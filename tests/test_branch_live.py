"""Opt-in live branch isolation test; never creates or alters MAIN objects.

Set PAS_TEST_BRANCH_LIVE=1 and PAS_TEST_HOST/PAS_TEST_USER/PAS_TEST_PASSWORD.
The account needs branch privileges and an existing user database. A unique
branch is created and dropped; no credentials or endpoints are stored here.
"""

import os
import uuid

import pytest
from server.config import ConnectionPoolConfig
from server.core.connection_cache import ConnectionCache
from server.core.sql_gateway import SQLGateway
from server.core.sql_executor import SQLExecutionError

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("PAS_TEST_BRANCH_LIVE") != "1", reason="requires explicit live branch test opt-in"
    ),
]


async def test_live_branch_isolation():
    cache = ConnectionCache(ConnectionPoolConfig())
    g = SQLGateway(cache)
    kw = dict(
        host=os.environ["PAS_TEST_HOST"],
        port=int(os.environ.get("PAS_TEST_PORT", "3306")),
        user=os.environ["PAS_TEST_USER"],
        password=os.environ["PAS_TEST_PASSWORD"],
        user_id="live-branch-test",
        instance_id="live-branch-test",
    )
    branch = "pas_verify_" + uuid.uuid4().hex[:12]
    table = "pas_probe_" + uuid.uuid4().hex[:12]
    created = False

    async def run(sql, branch=None, database=None):
        return await g.execute(**kw, sql=sql, branch=branch, database=database)

    try:
        before = await run("SHOW BRANCHES", branch="")
        assert not before.get("truncated"), "Complete initial branch inventory required"
        dbs = await run("SHOW DATABASES")
        candidates = [
            r[0]
            for r in dbs["rows"]
            if not r[0].startswith("__")
            and r[0].lower() not in {"mysql", "information_schema", "performance_schema", "sys", "__recycle_bin__"}
        ]
        if not candidates:
            raise RuntimeError("No existing user database available; no MAIN database will be created")
        db = candidates[0]
        created = True
        await run("CREATE BRANCH " + branch, branch="")
        print("PASS create disposable branch", branch, flush=True)
        await run(f"CREATE TABLE `{table}` (id INT PRIMARY KEY, note VARCHAR(40))", branch, db)
        await g.execute_transaction(
            **kw,
            database=db,
            branch=branch,
            sql_statements=[
                f"INSERT INTO `{table}` VALUES (1, 'branch only')",
                f"INSERT INTO `{table}` VALUES (2, 'committed')",
            ],
        )
        assert (await run(f"SELECT COUNT(*) FROM `{table}`", branch, db))["rows"][0][0] == 2
        print("PASS branch-local DDL and transaction commit", flush=True)
        try:
            await g.execute_transaction(
                **kw,
                database=db,
                branch=branch,
                sql_statements=[
                    f"INSERT INTO `{table}` VALUES (3, 'rollback')",
                    f"INSERT INTO `{table}` VALUES (1, 'duplicate')",
                ],
            )
        except SQLExecutionError:
            pass
        else:
            raise AssertionError("Expected duplicate key error")
        assert (await run(f"SELECT COUNT(*) FROM `{table}`", branch, db))["rows"][0][0] == 2
        print("PASS transaction rollback", flush=True)
        inv = await g.execute_parameterized(
            **kw, database=db, branch=branch, sql="SHOW TABLE STATUS LIKE %s", params=[table]
        )
        assert inv["rows"] and inv["rows"][0][0] == table
        cols = await g.execute_parameterized(
            **kw, database=db, branch=branch, sql=f"SHOW FULL COLUMNS FROM `{table}`", params=[]
        )
        assert [r[0] for r in cols["rows"]] == ["id", "note"]
        assert not (await run(f"SHOW TABLES LIKE '{table}'", "", db))["rows"]
        assert not (await run(f"SHOW TABLES LIKE '{table}'", None, db))["rows"]
        print("PASS branch metadata and explicit/implicit MAIN isolation", flush=True)
        await run("DROP BRANCH " + branch, branch="")
        created = False
        after = await run("SHOW BRANCHES", branch="")
        assert not after.get("truncated"), "Complete final branch inventory required"
        assert sorted(map(str, before["rows"])) == sorted(map(str, after["rows"]))
        print("PASS cleanup: branch inventory restored", flush=True)
    finally:
        try:
            if created:
                remaining = await run("SHOW BRANCHES", branch="")
                if remaining.get("truncated"):
                    raise AssertionError("Incomplete cleanup inventory for " + branch)
                index = [str(c).lower() for c in remaining["columns"]].index("branch")
                if any(row[index] == branch for row in remaining["rows"]):
                    await run("DROP BRANCH " + branch, branch="")
        finally:
            await cache.close_all()
