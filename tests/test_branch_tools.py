import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from server.core.binding_manager import UserSQLCredential
from server.core.responses import error_response
from server.core.sql_executor import RateLimitError, SQLExecutionError, reset_rate_limiters
from server.models import AuditStatus, Permission
from server.mcp.tools.branch_handler import (
    _BranchContext,
    _execute_branch_sql,
    _resolve_branch_context,
    handle_create_branch,
    handle_delete_branch,
    handle_list_branches,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def clean_rate_limiters():
    reset_rate_limiters()
    yield
    reset_rate_limiters()


def _user():
    return SimpleNamespace(id="uid", display_name="User")


def _branch_context():
    return _credential_context(SimpleNamespace(id="iid", name="inst"))


def _credential_context(target_instance, permission=Permission.READWRITE):
    credential = SimpleNamespace(
        username_ciphertext="encrypted-user",
        password_ciphertext="encrypted-password",
    )
    return _BranchContext(
        instance=target_instance,
        credential=credential,
        permission=permission,
        public_instance_id=target_instance.id,
        target_type="instance",
    )


def _resolved_credential(permission=Permission.READWRITE):
    return UserSQLCredential(
        binding=SimpleNamespace(),
        credential=SimpleNamespace(
            username_ciphertext="encrypted-user",
            password_ciphertext="encrypted-password",
        ),
        permission=permission,
    )


async def test_resolve_branch_context_returns_instance_resolution_error():
    user = SimpleNamespace(id="uid", display_name="User", status=None)
    session = object()
    expected = error_response("INSTANCE_NOT_ACCESSIBLE", "You don't have access to this instance.")

    with patch(
        "server.mcp.tools.branch_handler.resolve_target_instance",
        new_callable=AsyncMock,
        return_value=expected,
    ):
        result = await _resolve_branch_context(user, session, instance_id="iid")

    assert result == expected


async def test_resolve_branch_context_rejects_starting_instance_before_account_lookup():
    from server.models import InstanceStatus

    user = SimpleNamespace(id="uid", display_name="User", status=None)
    session = SimpleNamespace(execute=AsyncMock())
    target_instance = SimpleNamespace(id="iid", status=InstanceStatus.CREATING)

    with patch(
        "server.mcp.tools.branch_handler.resolve_target_instance",
        new_callable=AsyncMock,
        return_value=(target_instance, []),
    ):
        result = await _resolve_branch_context(user, session, instance_id="iid")

    session.execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {
        "error": "INSTANCE_STARTING",
        "message": "Instance is starting, please retry in a few seconds.",
    }


async def test_resolve_branch_context_wraps_account_creation_failure():
    from server.models import InstanceStatus, InstanceTopology

    user = SimpleNamespace(id="uid", display_name="User", status=None)
    result_proxy = SimpleNamespace(scalar_one_or_none=lambda: None)
    session = SimpleNamespace(execute=AsyncMock(return_value=result_proxy), commit=AsyncMock())
    target_instance = SimpleNamespace(
        id="iid",
        status=InstanceStatus.ACTIVE,
        topology=InstanceTopology.SINGLE_TENANT,
    )

    with (
        patch(
            "server.mcp.tools.branch_handler.resolve_target_instance",
            new_callable=AsyncMock,
            return_value=(target_instance, []),
        ),
        patch(
            "server.mcp.tools.branch_handler._resolve_user_sql_credential",
            new_callable=AsyncMock,
            return_value=error_response(
                "CONNECTION_ERROR",
                "Failed to create database account: create failed",
            ),
        ) as resolve_credential,
    ):
        result = await _resolve_branch_context(user, session, instance_id="iid")

    resolve_credential.assert_awaited_once()
    session.commit.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {
        "error": "CONNECTION_ERROR",
        "message": "Failed to create database account: create failed",
    }


@pytest.mark.parametrize("topology", ["SINGLE_TENANT", "MULTITENANT"])
async def test_branch_context_missing_credential_never_provisions(topology):
    from server.models import InstanceStatus, InstanceTopology

    user = SimpleNamespace(id="uid", display_name="User", status=None)
    session = AsyncMock()
    instance = SimpleNamespace(id="iid", status=InstanceStatus.ACTIVE,
                               topology=getattr(InstanceTopology, topology))
    with (
        patch("server.mcp.tools.branch_handler.resolve_target_instance",
              new=AsyncMock(return_value=(instance, []))),
        patch("server.mcp.tools.handlers.resolve_user_instance_access",
              new=AsyncMock(return_value=SimpleNamespace(permission=Permission.READWRITE))),
        patch("server.mcp.tools.handlers.get_user_credential", new=AsyncMock(return_value=None)),
        patch("server.mcp.tools.handlers.create_db_account", new_callable=AsyncMock) as create_account,
        patch("server.core.tenant_provisioner.ensure_tenant", new_callable=AsyncMock) as create_tenant,
    ):
        result = await _resolve_branch_context(user, session, instance_id="iid")
    assert json.loads(result["content"][0]["text"])["error"] == "CONNECTION_ERROR"
    create_account.assert_not_awaited()
    create_tenant.assert_not_awaited()
    session.commit.assert_not_awaited()
    session.add.assert_not_called()


async def test_list_branches_maps_show_branches_rows():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": ["Branch"], "rows": [["MAIN"], ["br1"]]}, _branch_context()),
    ) as execute, patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock):
        result = await handle_list_branches(_user(), object(), instance_id="iid")

    execute.assert_awaited_once()
    payload = json.loads(result["content"][0]["text"])
    assert payload == {
        "branches": [
            {"branch_name": "MAIN"},
            {"branch_name": "br1"},
        ],
    }
    assert execute.await_args.kwargs["max_rows"] == 10000


async def test_list_branches_exposes_truncated_result():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=(
            {"columns": ["Branch"], "rows": [["MAIN"]], "truncated": True},
            _branch_context(),
        ),
    ), patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock):
        result = await handle_list_branches(_user(), object(), instance_id="iid")

    payload = json.loads(result["content"][0]["text"])
    assert payload == {
        "branches": [{"branch_name": "MAIN"}],
        "truncated": True,
    }


async def test_list_branches_finds_branch_column_case_insensitively():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": ["Database", " branch "], "rows": [["app", "br1"]]}, _branch_context()),
    ), patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock):
        result = await handle_list_branches(_user(), object(), instance_id="iid")

    payload = json.loads(result["content"][0]["text"])
    assert payload == {"branches": [{"branch_name": "br1"}]}


async def test_list_branches_rejects_unexpected_columns():
    context = _branch_context()
    user = _user()
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": ["Database"], "rows": [["app"]]}, context),
    ), patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock) as log:
        result = await handle_list_branches(user, object(), instance_id="iid")

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "UNEXPECTED_RESULT"
    assert log.await_args.kwargs["status"] == AuditStatus.ERROR
    assert log.await_args.kwargs["error_message"] == "SHOW BRANCHES result does not include a Branch column."


async def test_list_branches_rejects_missing_column_metadata():
    context = _branch_context()
    user = _user()
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": [], "rows": [["MAIN"]]}, context),
    ), patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock) as log:
        result = await handle_list_branches(user, object(), instance_id="iid")

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "UNEXPECTED_RESULT"
    assert log.await_args.kwargs["status"] == AuditStatus.ERROR
    assert log.await_args.kwargs["error_message"] == "SHOW BRANCHES result does not include a Branch column."


async def test_list_branches_rejects_rows_missing_branch_column():
    context = _branch_context()
    user = _user()
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": ["Database", "Branch"], "rows": [["app"]]}, context),
    ), patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock) as log:
        result = await handle_list_branches(user, object(), instance_id="iid")

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "UNEXPECTED_RESULT"
    assert log.await_args.kwargs["status"] == AuditStatus.ERROR
    assert log.await_args.kwargs["error_message"] == "SHOW BRANCHES result row is missing the Branch column."


async def test_list_branches_logs_success_after_parsing():
    context = _branch_context()
    user = _user()
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": ["Branch"], "rows": [["MAIN"], ["br1"]]}, context),
    ), patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock) as log:
        result = await handle_list_branches(user, object(), instance_id="iid")

    payload = json.loads(result["content"][0]["text"])
    assert payload == {"branches": [{"branch_name": "MAIN"}, {"branch_name": "br1"}]}
    assert log.await_args.kwargs["status"] == AuditStatus.SUCCESS
    assert log.await_args.kwargs["row_count"] == 2


async def test_branch_tools_do_not_echo_instance_metadata():
    gateway_result = {
        "columns": ["Branch"],
        "rows": [["MAIN"]],
        "instance_id": "iid",
        "cluster_id": "cid",
    }
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=(gateway_result, _branch_context()),
    ), patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock):
        list_result = await handle_list_branches(_user(), object(), instance_id="iid")

    payload = json.loads(list_result["content"][0]["text"])
    assert payload == {"branches": [{"branch_name": "MAIN"}]}
    assert "instance_id" not in payload
    assert "cluster_id" not in payload

    command_result = {
        "columns": [],
        "rows": [],
        "instance_id": "iid",
        "cluster_id": "cid",
    }
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=(command_result, _branch_context()),
    ):
        create_result = await handle_create_branch(
            object(), object(), branch_name="br_new", instance_id="iid",
        )
        delete_result = await handle_delete_branch(
            object(), object(), branch_name="br_old", instance_id="iid",
        )

    for result in (create_result, delete_result):
        payload = json.loads(result["content"][0]["text"])
        assert "instance_id" not in payload
        assert "cluster_id" not in payload


async def test_execute_branch_sql_preserves_structured_sql_error():
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()
    target_instance = SimpleNamespace(id="iid", host="h", port=3306, name="inst")
    gateway = SimpleNamespace(execute=AsyncMock(side_effect=SQLExecutionError("Query timed out", "TIMEOUT")))

    with (
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
            return_value=_credential_context(target_instance),
        ),
        patch("server.mcp.tools.branch_handler.decrypt", return_value="pw"),
        patch("server.mcp.tools.branch_handler.get_gateway", return_value=gateway),
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock),
    ):
        result = await _execute_branch_sql(user, session, sql="CREATE BRANCH br1", action="create_branch")

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"error": "TIMEOUT", "message": "Query timed out"}
    assert "instance_id" not in payload
    assert "cluster_id" not in payload


async def test_execute_branch_sql_wraps_generic_connection_error():
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()
    target_instance = SimpleNamespace(id="iid", host="h", port=3306, name="inst")
    sentinel = "password=SECRET branch=SENSITIVE_BRANCH"
    gateway = SimpleNamespace(
        execute=AsyncMock(side_effect=RuntimeError(sentinel))
    )

    with (
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
            return_value=_credential_context(target_instance),
        ),
        patch("server.mcp.tools.branch_handler.decrypt", return_value="pw"),
        patch("server.mcp.tools.branch_handler.get_gateway", return_value=gateway),
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock) as log,
    ):
        result = await _execute_branch_sql(user, session, sql="CREATE BRANCH br1", action="create_branch")

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {
        "error": "INTERNAL_ERROR",
        "message": "Internal branch operation error",
    }
    assert "instance_id" not in payload
    assert "cluster_id" not in payload
    assert log.await_args.kwargs["status"] == AuditStatus.ERROR
    assert log.await_args.kwargs["error_code"] == "INTERNAL_ERROR"
    assert log.await_args.kwargs["error_message"] == (
        "Internal branch operation error"
    )
    assert sentinel not in str(log.await_args.kwargs)


async def test_execute_branch_sql_preserves_cancellation():
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()
    target_instance = SimpleNamespace(
        id="iid", host="h", port=3306, name="inst"
    )
    gateway = SimpleNamespace(
        execute=AsyncMock(side_effect=asyncio.CancelledError)
    )

    with (
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
            return_value=_credential_context(target_instance),
        ),
        patch("server.mcp.tools.branch_handler.decrypt", return_value="pw"),
        patch(
            "server.mcp.tools.branch_handler.get_gateway",
            return_value=gateway,
        ),
        patch(
            "server.mcp.tools.branch_handler.log_audit",
            new_callable=AsyncMock,
        ) as log,
        pytest.raises(asyncio.CancelledError),
    ):
        await _execute_branch_sql(
            user,
            session,
            sql="CREATE BRANCH br1",
            action="create_branch",
        )

    log.assert_not_awaited()


async def test_execute_branch_sql_rate_limits_before_resolving_context():
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()

    with (
        patch("server.mcp.tools.branch_handler._check_rate_limit", side_effect=RateLimitError()),
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
        ) as resolve_context,
        patch("server.mcp.tools.branch_handler.get_gateway") as get_gateway,
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock) as log,
    ):
        result = await _execute_branch_sql(
            user, session, sql="CREATE BRANCH br1", action="create_branch",
            instance_id="dbi-unresolved",
        )

    resolve_context.assert_not_awaited()
    get_gateway.assert_not_called()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"error": "RATE_LIMITED", "message": "Too many requests. Please slow down."}
    assert log.await_args.kwargs["status"] == AuditStatus.BLOCKED
    assert log.await_args.kwargs["error_message"] == "Rate limited"
    assert log.await_args.kwargs["target_id"] == "dbi-unresolved"
    assert "target_type" not in log.await_args.kwargs
    assert "instance_id" not in log.await_args.kwargs


async def test_execute_branch_sql_rejects_disabled_user_before_rate_limit():
    from server.models import UserStatus

    user = SimpleNamespace(id="uid", display_name="User", status=UserStatus.DISABLED)
    session = object()

    with (
        patch("server.mcp.tools.branch_handler._check_rate_limit", side_effect=RateLimitError()) as check_rate,
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
        ) as resolve_context,
        patch("server.mcp.tools.branch_handler.get_gateway") as get_gateway,
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock) as log,
    ):
        result = await _execute_branch_sql(user, session, sql="CREATE BRANCH br1", action="create_branch")

    check_rate.assert_not_called()
    resolve_context.assert_not_awaited()
    get_gateway.assert_not_called()
    log.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {
        "error": "USER_DISABLED",
        "message": "Your account has been disabled. Contact admin.",
    }


async def test_create_branch_rejects_disabled_user_before_identifier_validation():
    from server.models import UserStatus

    user = SimpleNamespace(id="uid", display_name="User", status=UserStatus.DISABLED)

    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            user, object(), branch_name="bad;name", include_databases=["also;bad"],
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {
        "error": "USER_DISABLED",
        "message": "Your account has been disabled. Contact admin.",
    }


async def test_delete_branch_rejects_disabled_user_before_identifier_validation():
    from server.models import UserStatus

    user = SimpleNamespace(id="uid", display_name="User", status=UserStatus.DISABLED)

    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_delete_branch(user, object(), branch_name="bad;name")

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {
        "error": "USER_DISABLED",
        "message": "Your account has been disabled. Contact admin.",
    }


async def test_execute_branch_sql_delegates_permissions_to_database():
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()
    target_instance = SimpleNamespace(id="iid", host="h", port=3306, name="inst")
    gateway = SimpleNamespace(execute=AsyncMock(return_value={"columns": [], "rows": [], "row_count": 0}))

    with (
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
            return_value=_credential_context(target_instance),
        ),
        patch("server.mcp.tools.branch_handler.decrypt", return_value="pw"),
        patch("server.mcp.tools.branch_handler.get_gateway", return_value=gateway),
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock),
    ):
        result = await _execute_branch_sql(user, session, sql="CREATE BRANCH br1", action="create_branch")

    assert "isError" not in result
    gateway.execute.assert_awaited_once()
    assert gateway.execute.await_args.kwargs["read_only"] is False
    assert gateway.execute.await_args.kwargs["branch"] == ""


@pytest.mark.parametrize("sql,action", [("CREATE BRANCH br1", "create_branch"), ("DROP BRANCH br1", "delete_branch")])
async def test_execute_branch_sql_blocks_writes_for_readonly_credential(sql, action):
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()
    target_instance = SimpleNamespace(
        id="iid", host="h", port=3306, name="inst"
    )
    gateway = SimpleNamespace(
        execute=AsyncMock(
            return_value={"columns": [], "rows": [], "row_count": 0}
        )
    )

    with (
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
            return_value=_credential_context(
                target_instance, Permission.READONLY
            ),
        ),
        patch(
            "server.mcp.tools.branch_handler.decrypt",
            return_value="credential",
        ),
        patch(
            "server.mcp.tools.branch_handler.get_gateway",
            return_value=gateway,
        ),
        patch(
            "server.mcp.tools.branch_handler.log_audit",
            new_callable=AsyncMock,
        ),
    ):
        result = await _execute_branch_sql(
            user, session, sql=sql, action=action,
        )

    gateway.execute.assert_not_awaited()
    assert json.loads(result["content"][0]["text"])["error"] == "READ_ONLY_ACCESS"


async def test_execute_branch_sql_marks_list_as_read_only():
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()
    target_instance = SimpleNamespace(id="iid", host="h", port=3306, name="inst")
    gateway = SimpleNamespace(execute=AsyncMock(return_value={
        "columns": ["Branch"],
        "rows": [("MAIN",)],
        "row_count": 1,
        "truncated": False,
    }))

    with (
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
            return_value=_credential_context(target_instance),
        ),
        patch("server.mcp.tools.branch_handler.decrypt", return_value="pw"),
        patch("server.mcp.tools.branch_handler.get_gateway", return_value=gateway),
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock),
    ):
        result = await _execute_branch_sql(
            user, session, sql="SHOW BRANCHES", action="list_branches",
            log_success=False, read_only=True,
        )

    assert result == (
        {"columns": ["Branch"], "rows": [("MAIN",)], "row_count": 1, "truncated": False},
        _credential_context(target_instance),
    )
    assert gateway.execute.await_args.kwargs["read_only"] is True
    assert gateway.execute.await_args.kwargs["branch"] == ""


async def test_execute_branch_sql_logs_success_audit_fields():
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()
    target_instance = SimpleNamespace(id="iid", host="h", port=3306, name="inst")
    gateway = SimpleNamespace(execute=AsyncMock(return_value={"columns": [], "rows": [], "row_count": 0}))

    with (
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
            return_value=_credential_context(target_instance),
        ),
        patch("server.mcp.tools.branch_handler.decrypt", return_value="pw"),
        patch("server.mcp.tools.branch_handler.get_gateway", return_value=gateway),
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock) as log,
    ):
        result = await _execute_branch_sql(user, session, sql="DROP BRANCH br1", action="delete_branch")

    assert result == ({"columns": [], "rows": [], "row_count": 0}, _credential_context(target_instance))
    assert log.await_args.kwargs["instance_id"] == "iid"
    assert log.await_args.kwargs["action"] == "delete_branch"
    assert "sql_text" not in log.await_args.kwargs
    assert log.await_args.kwargs["status"] == AuditStatus.SUCCESS
    assert log.await_args.kwargs["row_count"] == 0
    assert log.await_args.kwargs["user_name"] == "User"
    assert "instance_name" not in log.await_args.kwargs


async def test_execute_branch_sql_forces_default_branch_session():
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()
    target_instance = SimpleNamespace(id="iid", host="h", port=3306, name="inst")
    gateway = SimpleNamespace(execute=AsyncMock(return_value={"columns": [], "rows": [], "row_count": 0}))

    with (
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
            return_value=_credential_context(target_instance),
        ),
        patch("server.mcp.tools.branch_handler.decrypt", return_value="pw"),
        patch("server.mcp.tools.branch_handler.get_gateway", return_value=gateway),
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock),
    ):
        await _execute_branch_sql(user, session, sql="CREATE BRANCH br1", action="create_branch")

    gateway.execute.assert_awaited_once()
    assert gateway.execute.await_args.kwargs["database"] is None
    assert gateway.execute.await_args.kwargs["branch"] == ""


async def test_execute_branch_sql_can_skip_success_audit():
    user = SimpleNamespace(id="uid", display_name="User")
    session = object()
    target_instance = SimpleNamespace(id="iid", host="h", port=3306, name="inst")
    gateway = SimpleNamespace(execute=AsyncMock(return_value={"columns": ["Branch"], "rows": [["MAIN"]]}))

    with (
        patch(
            "server.mcp.tools.branch_handler._resolve_branch_context",
            new_callable=AsyncMock,
            return_value=_credential_context(target_instance),
        ),
        patch("server.mcp.tools.branch_handler.decrypt", return_value="pw"),
        patch("server.mcp.tools.branch_handler.get_gateway", return_value=gateway),
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock) as log,
    ):
        result = await _execute_branch_sql(
            user, session, sql="SHOW BRANCHES", action="list_branches",
            log_success=False,
        )

    assert result == ({"columns": ["Branch"], "rows": [["MAIN"]]}, _credential_context(target_instance))
    log.assert_not_awaited()


async def test_create_branch_generates_create_branch_sql():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": [], "rows": []}, object()),
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="br_new",
            include_databases=None,
            instance_id="iid",
        )

    assert execute.await_args.kwargs["sql"] == "CREATE BRANCH br_new"
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"branch_name": "br_new", "status": "created"}


async def test_create_branch_empty_include_databases_omits_with_clause():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": [], "rows": []}, object()),
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="br_new",
            include_databases=[],
            instance_id="iid",
        )

    assert execute.await_args.kwargs["sql"] == "CREATE BRANCH br_new"
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"branch_name": "br_new", "status": "created"}


async def test_create_branch_preserves_branch_name_before_sql_and_output():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": [], "rows": []}, object()),
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="Br_New",
            include_databases=None,
            instance_id="iid",
        )

    assert execute.await_args.kwargs["sql"] == "CREATE BRANCH Br_New"
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"branch_name": "Br_New", "status": "created"}


async def test_create_branch_generates_with_database_sql():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": [], "rows": []}, object()),
    ) as execute:
        await handle_create_branch(
            object(), object(),
            branch_name="br_new",
            include_databases=["db1", "db2"],
            instance_id="iid",
        )

    assert execute.await_args.kwargs["sql"] == "CREATE BRANCH br_new WITH DATABASE db1, db2"


async def test_create_branch_rejects_minimal_injection_chars():
    result = await handle_create_branch(
        object(), object(),
        branch_name="br1;DROP",
        include_databases=None,
    )

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


@pytest.mark.parametrize(
    "branch_name",
    [
        "br,other",
        "br`x",
        "br'x",
        'br"x',
        "br#x",
        "br/*x*/",
        "br--x",
        "br x",
    ],
)
async def test_create_branch_rejects_structural_identifier_chars_before_sql(branch_name):
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name=branch_name,
            include_databases=None,
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


async def test_create_branch_rejects_empty_branch_name_before_sql():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="",
            include_databases=None,
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


async def test_create_branch_rejects_overlong_branch_name_before_sql():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="b" * 257,
            include_databases=None,
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


async def test_create_branch_rejects_non_string_branch_name_before_sql():
    invalid_branch_name: Any = 123
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name=invalid_branch_name,
            include_databases=None,
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


async def test_create_branch_rejects_invalid_include_database_before_sql():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="br1",
            include_databases=["db1", "bad;db"],
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


@pytest.mark.parametrize(
    "database_name",
    [
        "db,other",
        "db`x",
        "db'x",
        'db"x',
        "db#x",
        "db/*x*/",
        "db--x",
        "db x",
    ],
)
async def test_create_branch_rejects_structural_include_database_chars_before_sql(database_name):
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="br1",
            include_databases=["db1", database_name],
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


async def test_create_branch_rejects_non_list_include_databases_before_sql():
    invalid_include_databases: Any = "db1"
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="br1",
            include_databases=invalid_include_databases,
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


async def test_create_branch_rejects_blank_include_database_before_sql():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="br1",
            include_databases=["db1", "  "],
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


async def test_create_branch_rejects_non_string_include_database_before_sql():
    invalid_include_databases: Any = ["db1", 123]
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_create_branch(
            object(), object(),
            branch_name="br1",
            include_databases=invalid_include_databases,
        )

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


async def test_create_branch_returns_sql_error():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=error_response("SQL_ERROR", "Branch already exists"),
    ):
        result = await handle_create_branch(
            object(), object(),
            branch_name="br_existing",
            include_databases=None,
        )

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"error": "SQL_ERROR", "message": "Branch already exists"}


async def test_create_branch_preserves_structured_sql_error():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=error_response("SQL_ERROR", "Branch already exists"),
    ):
        result = await handle_create_branch(
            object(), object(),
            branch_name="br_existing",
            include_databases=None,
        )

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"error": "SQL_ERROR", "message": "Branch already exists"}


async def test_create_branch_timeout_returns_unknown_status():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=error_response(
            "TIMEOUT",
            "Query timed out",
            instance_id="iid",
            cluster_id="cid",
        ),
    ):
        result = await handle_create_branch(
            object(), object(),
            branch_name="br_slow",
            include_databases=None,
        )

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "OPERATION_STATUS_UNKNOWN"
    assert payload["branch_name"] == "br_slow"
    assert "list_branches" in payload["message"]
    assert "instance_id" not in payload
    assert "cluster_id" not in payload


async def test_delete_branch_generates_drop_branch_sql():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": [], "rows": []}, object()),
    ) as execute:
        result = await handle_delete_branch(
            object(), object(),
            branch_name="br_old",
            instance_id="iid",
        )

    assert execute.await_args.kwargs["sql"] == "DROP BRANCH br_old"
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"branch_name": "br_old", "status": "deleted"}


async def test_delete_branch_preserves_branch_name_before_sql_and_output():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": [], "rows": []}, object()),
    ) as execute:
        result = await handle_delete_branch(
            object(), object(),
            branch_name="Br_Old",
            instance_id="iid",
        )

    assert execute.await_args.kwargs["sql"] == "DROP BRANCH Br_Old"
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"branch_name": "Br_Old", "status": "deleted"}


@pytest.mark.parametrize("branch_name", ["MAIN", "main"])
async def test_delete_branch_delegates_main_branch_to_sql(branch_name):
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=({"columns": [], "rows": []}, object()),
    ) as execute:
        result = await handle_delete_branch(object(), object(), branch_name=branch_name)

    assert execute.await_args.kwargs["sql"] == f"DROP BRANCH {branch_name}"
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"branch_name": branch_name, "status": "deleted"}


async def test_delete_branch_rejects_empty_branch_name_before_sql():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
    ) as execute:
        result = await handle_delete_branch(object(), object(), branch_name="")

    execute.assert_not_awaited()
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "INVALID_IDENTIFIER"


async def test_delete_branch_returns_sql_error():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=error_response("SQL_ERROR", "Branch does not exist"),
    ):
        result = await handle_delete_branch(
            object(), object(),
            branch_name="br_missing",
        )

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"error": "SQL_ERROR", "message": "Branch does not exist"}


@pytest.mark.parametrize("inventory,expected,calls", [
    ({"columns": ["Branch"], "rows": [["MAIN"]]}, "not_found", 1),
    ({"columns": ["Branch"], "rows": [["br"]]}, "deleted", 2),
    ({"columns": ["Branch"], "rows": [["BR"]]}, "not_found", 1),
    ({"columns": ["Branch"], "rows": [], "truncated": True}, "UNEXPECTED_RESULT", 1),
    ({"columns": [], "rows": []}, "UNEXPECTED_RESULT", 1),
])
async def test_delete_branch_checks_complete_inventory(inventory, expected, calls):
    instance = SimpleNamespace(id="iid", host="h", port=3306)
    gateway = SimpleNamespace(execute=AsyncMock(side_effect=[inventory, {"row_count": 0}]))
    with (
        patch("server.mcp.tools.branch_handler._resolve_branch_context", new_callable=AsyncMock,
              return_value=_credential_context(instance)),
        patch("server.mcp.tools.branch_handler.decrypt", return_value="pw"),
        patch("server.mcp.tools.branch_handler.get_gateway", return_value=gateway),
        patch("server.mcp.tools.branch_handler.log_audit", new_callable=AsyncMock),
    ):
        result = await handle_delete_branch(_user(), object(), branch_name="br")
    payload = json.loads(result["content"][0]["text"])
    assert payload.get("error", payload.get("status")) == expected
    assert gateway.execute.await_count == calls
    assert gateway.execute.await_args_list[0].kwargs["sql"] == "SHOW BRANCHES"


async def test_lost_connection_reports_uncertain_outcome_without_native_details():
    from server.core.sql_gateway import _sanitized_sql_error

    error = _sanitized_sql_error(Exception(2013, "private database endpoint"))
    assert error.code == "DB_CONNECTION_LOST"
    assert "verify" in error.message.lower()
    assert "private" not in error.message


async def test_delete_branch_timeout_returns_unknown_status():
    with patch(
        "server.mcp.tools.branch_handler._execute_branch_sql",
        new_callable=AsyncMock,
        return_value=error_response(
            "TIMEOUT",
            "Query timed out",
            instance_id="iid",
            cluster_id="cid",
        ),
    ):
        result = await handle_delete_branch(
            object(), object(),
            branch_name="br_slow_drop",
        )

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"] == "OPERATION_STATUS_UNKNOWN"
    assert payload["branch_name"] == "br_slow_drop"
    assert "list_branches" in payload["message"]
    assert "instance_id" not in payload
    assert "cluster_id" not in payload
