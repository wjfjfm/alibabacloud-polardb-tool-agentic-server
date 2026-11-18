"""Opt-in authenticated MCP acceptance with real SQL connections.

Requires the same PAS_TEST_* variables as test_branch_live. Optionally set
PAS_TEST_NO_BRANCH_PORT (and HOST/USER/PASSWORD with that prefix) for a separate
local endpoint without branch support. Only disposable branches are mutated.
"""

import asyncio
import json
import os
import uuid

import pytest
from sqlalchemy import select

from server.config import ConnectionPoolConfig, get_config
from server.core import admin_binding_service, personal_token_service
from server.core.connection_cache import ConnectionCache
from server.core.crypto import encrypt
from server.core.sql_gateway import SQLGateway
from server.db import engine as engine_mod
from server.mcp.tools import reset_gateway, set_gateway
from server.models import (
    AgentInstanceBinding, AgentInstanceBindingCapability, AllocationMode,
    BindingCapability, CredentialCapability, CredentialPurpose, Instance,
    InstanceCredential, InstanceStatus, InstanceTopology, Permission,
    UserInstanceBinding,
)
from tests.test_mcp_transport import (
    MCP_HEADERS, _create_mcp_token, _jsonrpc, _parse_sse_response,
    clean as clean, client as client, encryption_key as encryption_key,
    setup_data as setup_data, test_engine as test_engine,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(os.environ.get("PAS_TEST_BRANCH_LIVE") != "1",
                       reason="requires explicit live branch test opt-in"),
]


@pytest.mark.parametrize("mode", ["workspace", "pinned", "personal"])
async def test_authenticated_branch_lifecycle(client, setup_data, encryption_key, mode):
    personal = mode == "personal"
    factory = engine_mod.get_session_factory()
    instance_id = setup_data["instance"].id
    user = setup_data["admin"]
    agent = setup_data["agent"]
    get_config().server.public_base_url = "http://localhost:18760"
    # Exercise a lifecycle in one burst without changing production defaults.
    get_config().sql_security.rate_limit.burst = 100
    async with factory() as session:
        instance = await session.get(Instance, instance_id)
        instance.host = os.environ["PAS_TEST_HOST"]
        instance.port = int(os.environ.get("PAS_TEST_PORT", "3306"))
        credential = await session.scalar(select(InstanceCredential).where(
            InstanceCredential.instance_id == instance_id))
        credential.username_ciphertext = encrypt(os.environ["PAS_TEST_USER"], key=encryption_key)
        credential.password_ciphertext = encrypt(os.environ["PAS_TEST_PASSWORD"], key=encryption_key)
        if personal:
            await admin_binding_service.update_user_instance_access(
                session, user_id=user.id, instance_id=instance_id, credential_id=credential.id,
                permission=Permission.READWRITE,
                capabilities={BindingCapability.SQL_READ, BindingCapability.SQL_WRITE}, enabled=True)
            _, token = await personal_token_service.issue(
                session, user.id, resource=get_config().server.public_base_url + "/mcp/personal")
        else:
            token = _create_mcp_token(
                f"user:{user.id}", user.credential_epoch, agent.id if mode == "pinned" else None)
        await session.commit()

    cache = ConnectionCache(ConnectionPoolConfig())
    set_gateway(SQLGateway(cache))
    endpoint = "/mcp/personal" if personal else "/mcp"
    headers = {**MCP_HEADERS, "Authorization": f"Bearer {token}"}

    async def call(name, arguments, *, ok=True):
        response = await client.post(endpoint, headers=headers, json=_jsonrpc(
            "tools/call", {"name": name, "arguments": {"instance_id": instance_id, **arguments}}))
        assert response.status_code == 200, response.text
        result = _parse_sse_response(response.text)[0]["result"]
        assert bool(result.get("isError")) is not ok, result
        return json.loads(result["content"][0]["text"])

    branch = "pas_mcp_" + uuid.uuid4().hex[:12]
    table = "pas_probe_" + uuid.uuid4().hex[:12]
    created = False
    try:
        listed = await client.post(endpoint, headers=headers, json=_jsonrpc("tools/list"))
        names = {t["name"] for t in _parse_sse_response(listed.text)[0]["result"]["tools"]}
        assert {"list_branches", "create_branch", "delete_branch"} <= names
        before = await call("list_branches", {})
        databases = await call("run_sql", {"sql": "SHOW DATABASES"})
        candidates = [r[0] for r in databases["rows"] if not r[0].startswith("__")
                      and r[0].lower() not in {"mysql", "sys", "information_schema", "performance_schema"}]
        assert candidates, "An existing user database is required; MAIN will not be modified"
        db = candidates[0]
        created = True
        await call("create_branch", {"branch_name": branch})
        args = {"branch": branch, "database": db}
        await call("run_sql", {**args, "sql": f"CREATE TABLE `{table}` (id INT PRIMARY KEY)"})
        await call("run_sql_transaction", {**args, "sql_statements": [
            f"INSERT INTO `{table}` VALUES (1)", f"INSERT INTO `{table}` VALUES (2)"]})
        await call("run_sql_transaction", {**args, "sql_statements": [
            f"INSERT INTO `{table}` VALUES (3)", f"INSERT INTO `{table}` VALUES (1)"]}, ok=False)
        rows = await call("run_sql", {**args, "sql": f"SELECT id FROM `{table}` ORDER BY id"})
        assert rows["rows"] == [[1], [2]]
        schema = await call("describe_schema", {**args, "table_pattern": table})
        assert len(schema["tables"]) == 1, schema
        main, isolated = await asyncio.gather(
            call("run_sql", {"database": db, "sql": f"SHOW TABLES LIKE '{table}'"}),
            call("run_sql", {**args, "sql": f"SELECT COUNT(*) FROM `{table}`"}),
        )
        assert not main["rows"]
        assert isolated["rows"] == [[2]]
        await call("run_sql", {**args, "branch": branch + "_absent", "sql": "SELECT 1"}, ok=False)
        await call("run_sql", {**args, "sql": "SET @@session.branch = ''"}, ok=False)
        await call("run_sql", {**args, "sql": "PREPARE s FROM 'SELECT 1'"}, ok=False)

        # A different endpoint must not inherit the remote branch or connection.
        if os.environ.get("PAS_TEST_NO_BRANCH_PORT"):
            async with factory() as session:
                other = Instance(cluster_id="local-negative", name="Local negative endpoint",
                                 topology=InstanceTopology.SINGLE_TENANT,
                                 allocation_mode=AllocationMode.REGISTERED,
                                 host=os.environ.get("PAS_TEST_NO_BRANCH_HOST", "127.0.0.1"),
                                 port=int(os.environ["PAS_TEST_NO_BRANCH_PORT"]), status=InstanceStatus.ACTIVE)
                session.add(other)
                await session.flush()
                cred = InstanceCredential(instance_id=other.id, name="test-local",
                    purpose=CredentialPurpose.DIRECT_ACCESS, capability=CredentialCapability.READWRITE,
                    username_ciphertext=encrypt(os.environ.get("PAS_TEST_NO_BRANCH_USER", "root"), key=encryption_key),
                    password_ciphertext=encrypt(os.environ.get("PAS_TEST_NO_BRANCH_PASSWORD", ""), key=encryption_key),
                    created_by_user_id=user.id)
                session.add(cred)
                await session.flush()
                await admin_binding_service.update_user_instance_access(
                    session, user_id=user.id, instance_id=other.id, credential_id=cred.id,
                    permission=Permission.READWRITE, capabilities={BindingCapability.SQL_READ, BindingCapability.SQL_WRITE},
                    enabled=True)
                binding = AgentInstanceBinding(agent_id=agent.id, instance_id=other.id,
                    credential_id=cred.id, permission=Permission.READWRITE, created_by_user_id=user.id)
                binding.capabilities = [AgentInstanceBindingCapability(capability=c) for c in
                                        (BindingCapability.SQL_READ, BindingCapability.SQL_WRITE)]
                session.add(binding)
                await session.commit()
                other_id = other.id
            local = await call("run_sql", {"instance_id": other_id, "sql": "SELECT @@port"})
            assert local["rows"][0][0] == int(os.environ["PAS_TEST_NO_BRANCH_PORT"])
            await call("list_branches", {"instance_id": other_id}, ok=False)
            await call("run_sql", {"instance_id": other_id, "branch": branch, "sql": "SELECT 1"}, ok=False)
            await call("run_sql", {"instance_id": other_id, "sql": "SELECT 1"})
            assert (await call("run_sql", {**args, "sql": f"SELECT COUNT(*) FROM `{table}`"}))["rows"] == [[2]]

        # Change only the existing grant: read access survives, mutations fail closed.
        async with factory() as session:
            cls = UserInstanceBinding if personal else AgentInstanceBinding
            binding = await session.scalar(select(cls).where(cls.instance_id == instance_id))
            binding.permission = Permission.READONLY
            await session.commit()
        await call("list_branches", {})
        await call("run_sql", {**args, "sql": f"SELECT * FROM `{table}`"})
        for name, arguments in [
            ("create_branch", {"branch_name": branch}),
            ("delete_branch", {"branch_name": branch}),
            ("run_sql", {**args, "sql": f"INSERT INTO `{table}` VALUES (4)"}),
        ]:
            result = await call(name, arguments, ok=False)
            assert "READ_ONLY" in json.dumps(result), result
        async with factory() as session:
            binding = await session.scalar(select(cls).where(cls.instance_id == instance_id))
            binding.permission = Permission.READWRITE
            await session.commit()
        await call("delete_branch", {"branch_name": branch})
        created = False
        assert (await call("delete_branch", {"branch_name": branch}))["status"] == "not_found"
        assert await call("list_branches", {}) == before
    finally:
        try:
            if created:
                # Cleanup bypasses a deliberately reduced application grant, not DB auth.
                await SQLGateway(cache).execute(host=os.environ["PAS_TEST_HOST"],
                    port=int(os.environ.get("PAS_TEST_PORT", "3306")),
                    user=os.environ["PAS_TEST_USER"], password=os.environ["PAS_TEST_PASSWORD"],
                    user_id="live-cleanup", instance_id=instance_id, database=None,
                    branch="", sql=f"DROP BRANCH {branch}")
        finally:
            reset_gateway()
            await cache.close_all()
