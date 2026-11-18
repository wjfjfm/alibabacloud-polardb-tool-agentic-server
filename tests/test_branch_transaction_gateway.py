"""Adapted bcb3570: branch transactions must never leak pooled session state."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.core.sql_gateway import SQLGateway


@pytest.mark.asyncio
@pytest.mark.parametrize("statement", [
    "PREPARE s FROM 'SET @@branch = MAIN'", "EXECUTE s",
    "DEALLOCATE PREPARE s", "CALL switch_branch()", "/*!50000 EXECUTE s */",
])
async def test_branch_transaction_rejects_indirect_session_mutation(statement):
    from server.mcp.tools.handlers import handle_run_sql_transaction

    result = await handle_run_sql_transaction(
        SimpleNamespace(id="user", status="active"), AsyncMock(),
        sql_statements=[statement], branch="isolated",
    )
    assert result["isError"] is True
    assert "BLOCKED_SQL" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "branch,statement",
    [
        ("isolated", "SELECT 1"),
        ("", "SELECT 1"),
        (None, "CALL changes_branch()"),
        (None, "/*!50000 EXECUTE prepared */"),
    ],
)
async def test_transaction_discards_branch_and_dynamic_sessions(branch, statement):
    cursor = AsyncMock()
    cursor.description = None
    cursor.fetchall.return_value = []
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=cursor)
    context.__aexit__ = AsyncMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = context
    cache = AsyncMock()
    cache.acquire.return_value = conn
    gateway = SQLGateway(cache)
    await gateway.execute_transaction(
        host="host",
        port=3306,
        user="user",
        password="secret",
        sql_statements=[statement],
        database=None,
        user_id="u",
        instance_id="i",
        branch=branch,
    )
    if branch is not None:
        calls = cursor.execute.call_args_list
        branch_index = next(i for i, call in enumerate(calls) if call.args == ("SET @@session.branch = %s", (branch,)))
        begin_index = next(i for i, call in enumerate(calls) if call.args[0] == "BEGIN")
        assert branch_index < begin_index
    assert cache.release.call_args.kwargs["discard"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["execute", "execute_transaction", "execute_parameterized"])
async def test_branch_selects_database_after_switch_and_discards(method):
    cursor = AsyncMock()
    cursor.description = None
    cursor.fetchall.return_value = []
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=cursor)
    context.__aexit__ = AsyncMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = context
    cache = AsyncMock()
    cache.acquire.return_value = conn
    kwargs = dict(
        host="h", port=3306, user="u", password="p", database="db`name", user_id="u", instance_id="i", branch="isolated"
    )
    if method == "execute_transaction":
        kwargs["sql_statements"] = ["SELECT 1"]
    else:
        kwargs["sql"] = "SELECT 1"
        if method == "execute_parameterized":
            kwargs["params"] = []
    await getattr(SQLGateway(cache), method)(**kwargs)
    calls = [call.args[0] for call in cursor.execute.call_args_list]
    assert calls.index("SET @@session.branch = %s") < calls.index("USE `db``name`")
    assert calls.index("USE `db``name`") < calls.index(
        "SELECT 1" if method != "execute" else next(s for s in calls if s.startswith("SELECT"))
    )
    assert cache.release.call_args.kwargs["discard"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_parameterized_branch_selection_is_bound_and_connection_discarded(fail):
    from server.core.sql_executor import SQLExecutionError

    cursor = AsyncMock()
    cursor.description = None
    cursor.fetchall.return_value = []
    cursor.fetchmany.return_value = [(i,) for i in range(10001)]
    if fail:
        cursor.execute.side_effect = RuntimeError("database detail")
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=cursor)
    context.__aexit__ = AsyncMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = context
    cache = AsyncMock()
    cache.acquire.return_value = conn
    gateway = SQLGateway(cache)
    kwargs = dict(
        host="host",
        port=3306,
        user="user",
        password="secret",
        sql="SHOW TABLE STATUS",
        params=[],
        database="db",
        user_id="u",
        instance_id="i",
        branch="isolated",
    )
    if fail:
        with pytest.raises(SQLExecutionError):
            await gateway.execute_parameterized(**kwargs)
    else:
        result = await gateway.execute_parameterized(**kwargs)
        assert result["truncated"] is True
        assert len(result["rows"]) == 10000
        cursor.fetchmany.assert_awaited_once_with(10001)
    assert cursor.execute.call_args_list[0].args == ("SET @@session.branch = %s", ("isolated",))
    assert cache.release.call_args.kwargs["discard"] is True
