from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from server.auth.principal import PrincipalKind
from server.mcp.authorized_server import (
    AGENT_SQL_TOOL_NAMES,
    BRANCH_TOOL_NAMES,
    AuthorizedFastMCP,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def catalog(monkeypatch):
    import server.mcp.authorized_server as module

    monkeypatch.setattr("server.features.knowledge._runtime", None)
    monkeypatch.setattr(module, "get_access_token", lambda: SimpleNamespace())
    monkeypatch.setattr(module, "_is_agent_user_token", lambda token: False)
    personal = {"value": True}
    monkeypatch.setattr(module, "is_personal_access", lambda token: personal["value"])
    user = SimpleNamespace(kind=PrincipalKind.USER)
    policy = AsyncMock(return_value=(user, user, AGENT_SQL_TOOL_NAMES | BRANCH_TOOL_NAMES))
    monkeypatch.setattr(module, "_request_catalog_policy", policy)
    mcp = AuthorizedFastMCP("catalog-test")
    for name in AGENT_SQL_TOOL_NAMES | BRANCH_TOOL_NAMES:

        async def operation(instance_id: str | None = None, branch: str | None = None):
            return "ok"

        mcp.add_tool(operation, name=name)
    return mcp, policy, personal, user


async def test_personal_agent_personal_schemas_are_request_local(catalog):
    mcp, policy, personal, user = catalog
    first = {tool.name: tool for tool in await mcp.list_tools()}
    assert BRANCH_TOOL_NAMES <= first.keys()
    for name in AGENT_SQL_TOOL_NAMES | BRANCH_TOOL_NAMES:
        assert "instance_id" in first[name].inputSchema["required"]
    personal["value"] = False
    policy.return_value = (user, SimpleNamespace(kind=PrincipalKind.AGENT), AGENT_SQL_TOOL_NAMES | BRANCH_TOOL_NAMES)
    agent = {tool.name: tool for tool in await mcp.list_tools()}
    assert BRANCH_TOOL_NAMES <= agent.keys()
    for name in AGENT_SQL_TOOL_NAMES:
        assert "branch" in agent[name].inputSchema["properties"]
    personal["value"] = True
    policy.return_value = (user, user, AGENT_SQL_TOOL_NAMES | BRANCH_TOOL_NAMES)
    again = {tool.name: tool for tool in await mcp.list_tools()}
    for name in AGENT_SQL_TOOL_NAMES:
        assert "branch" in again[name].inputSchema["properties"]
        assert again[name].inputSchema == first[name].inputSchema


@pytest.mark.parametrize("name", sorted(BRANCH_TOOL_NAMES))
async def test_workspace_branch_call_dispatches_to_registered_handler(catalog, name):
    mcp, policy, personal, user = catalog
    personal["value"] = False
    policy.return_value = (user, SimpleNamespace(kind=PrincipalKind.AGENT), frozenset())
    result = await mcp.call_tool(name, {"instance_id": "instance"})
    assert result is not None


async def test_personal_without_sql_grant_hides_branch_tools(catalog):
    mcp, policy, _, user = catalog
    policy.return_value = (user, user, frozenset())
    assert not await mcp.list_tools()
    policy.return_value = (user, None, frozenset())
    assert not await mcp.list_tools()


@pytest.mark.parametrize("name", sorted(AGENT_SQL_TOOL_NAMES | BRANCH_TOOL_NAMES))
async def test_personal_calls_require_explicit_instance_before_dispatch(catalog, name):
    mcp, *_ = catalog
    with pytest.raises(ToolError, match="requires instance_id"):
        await mcp.call_tool(name, {})
