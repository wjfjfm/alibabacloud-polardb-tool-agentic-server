from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from server.core.audit_logger import log_audit
from server.core.crypto import decrypt
from server.core.responses import error_response
from server.core.sql_executor import RateLimitError, SQLExecutionError, _check_rate_limit
from server.models import (
    AuditStatus,
    Instance,
    InstanceCredential,
    InstanceStatus,
    Permission,
    UserStatus,
)
from server.mcp.tools import get_gateway, resolve_target_instance
from server.mcp.tools.agent_sql_access import resolve_agent_sql_access
from server.mcp.tools.handlers import (
    SQLActor, _actor_audit_fields, _actor_cache_key, _is_agent_actor,
    _resource_agent, _resolve_user_sql_credential,
)
from server.mcp.tools.identifier import validate_identifier_minimal


_BRANCH_LIST_MAX_ROWS = 10000


@dataclass(frozen=True)
class _BranchContext:
    instance: Instance
    credential: InstanceCredential
    permission: Permission
    public_instance_id: str
    target_type: Literal["instance", "db_instance_resource"]


def _branch_audit_fields(actor: SQLActor, context: _BranchContext) -> dict:
    return {
        **_actor_audit_fields(actor),
        "instance_id": context.instance.id,
        "target_id": context.public_instance_id,
        "target_type": context.target_type,
    }


def _disabled_user_error(user: object) -> dict | None:
    if getattr(user, "status", None) == UserStatus.DISABLED:
        return error_response("USER_DISABLED", "Your account has been disabled. Contact admin.")
    return None


def _branch_column_index(columns: list) -> int | None:
    for index, column in enumerate(columns):
        if str(column).strip().lower() == "branch":
            return index
    return None


def _error_payload(result: dict) -> dict | None:
    if not result.get("isError"):
        return None
    try:
        payload = json.loads(result["content"][0]["text"])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _branch_timeout_unknown(operation: str, branch_name: str) -> dict:
    return error_response(
        "OPERATION_STATUS_UNKNOWN",
        (
            f"{operation} timed out. The operation may still complete on PolarDB; "
            f"call list_branches to verify whether branch '{branch_name}' exists."
        ),
        branch_name=branch_name,
    )


async def _resolve_branch_context(
    user: SQLActor,
    session: AsyncSession,
    *,
    instance_id: str | None = None,
    session_factory=None,
    background_tasks=None,
) -> _BranchContext | dict:
    if _is_agent_actor(user):
        access = await resolve_agent_sql_access(
            _resource_agent(user), session, instance_id=instance_id, database=None,
        )
        if isinstance(access, dict):
            return access
        if access.instance.status == InstanceStatus.CREATING:
            return error_response("INSTANCE_STARTING", "Instance is starting, please retry in a few seconds.")
        return _BranchContext(
            instance=access.instance,
            credential=access.credential,
            permission=access.permission,
            public_instance_id=access.public_instance_id,
            target_type="db_instance_resource" if access.source == "provisioned" else "instance",
        )

    if user.status == UserStatus.DISABLED:
        return error_response("USER_DISABLED", "Your account has been disabled. Contact admin.")

    result = await resolve_target_instance(
        user, session, instance_id,
        session_factory=session_factory, background_tasks=background_tasks,
    )
    if isinstance(result, dict):
        return result
    target_instance, _accessible = result

    if target_instance.status == InstanceStatus.CREATING:
        return error_response("INSTANCE_STARTING", "Instance is starting, please retry in a few seconds.")

    credential_result = await _resolve_user_sql_credential(
        user, target_instance, session, allow_provisioning=False,
    )
    if isinstance(credential_result, dict):
        return credential_result
    return _BranchContext(
        instance=target_instance,
        credential=credential_result.credential,
        permission=credential_result.permission,
        public_instance_id=target_instance.id,
        target_type="instance",
    )


async def _execute_branch_sql(
    user: SQLActor,
    session: AsyncSession,
    *,
    sql: str,
    action: str,
    max_rows: int = 1000,
    instance_id: str | None = None,
    session_factory=None,
    background_tasks=None,
    log_success: bool = True,
    read_only: bool = False,
    delete_branch_name: str | None = None,
) -> tuple[dict, _BranchContext] | dict:
    if getattr(user, "status", None) == UserStatus.DISABLED:
        return error_response("USER_DISABLED", "Your account has been disabled. Contact admin.")

    try:
        _check_rate_limit(_actor_cache_key(user))
    except RateLimitError:
        # The requested ID has not been resolved or authorized yet.
        await log_audit(
            session, **_actor_audit_fields(user), target_id=instance_id, action=action,
            status=AuditStatus.BLOCKED,
            error_message="Rate limited",
            error_code="RATE_LIMITED",
        )
        return error_response("RATE_LIMITED", "Too many requests. Please slow down.")

    context = await _resolve_branch_context(
        user, session, instance_id=instance_id,
        session_factory=session_factory, background_tasks=background_tasks,
    )
    if isinstance(context, dict):
        return context
    target_instance = context.instance
    credential = context.credential
    permission = context.permission
    if permission == Permission.READONLY and action != "list_branches":
        await log_audit(
            session, **_branch_audit_fields(user, context),
            action=action, status=AuditStatus.BLOCKED,
            error_message="Branch mutation rejected for READONLY principal",
            error_code="READ_ONLY_ACCESS",
        )
        return error_response(
            "READ_ONLY_ACCESS", "Read-only access cannot create or delete branches."
        )
    read_only = read_only or permission == Permission.READONLY
    if (
        credential.username_ciphertext is None
        or credential.password_ciphertext is None
    ):
        return error_response(
            "CONNECTION_ERROR", "Database credential is unavailable."
        )
    username = decrypt(credential.username_ciphertext)
    password = decrypt(credential.password_ciphertext)
    start_time = time.time()
    async def execute(statement: str, limit: int) -> dict:
        return await get_gateway().execute(
            host=target_instance.host or "",
            port=target_instance.port or 3306,
            user=username,
            password=password,
            sql=statement,
            database=None,
            max_rows=limit,
            offset=0,
            user_id=_actor_cache_key(user),
            instance_id=context.public_instance_id,
            read_only=read_only,
            branch="",
        )

    try:
        missing = False
        if delete_branch_name is not None:
            inventory = await execute("SHOW BRANCHES", _BRANCH_LIST_MAX_ROWS)
            index = _branch_column_index(inventory.get("columns", []))
            rows = inventory.get("rows", [])
            if index is None or any(len(row) <= index for row in rows):
                raise SQLExecutionError("Invalid SHOW BRANCHES result.", "UNEXPECTED_RESULT")
            missing = not any(row[index] == delete_branch_name for row in rows)
            if missing and inventory.get("truncated"):
                raise SQLExecutionError(
                    "Branch inventory is incomplete; deletion was not attempted.",
                    "UNEXPECTED_RESULT",
                )
        result_data = {"status": "not_found"} if missing else await execute(sql, max_rows)
    except SQLExecutionError as e:
        await log_audit(
            session, **_branch_audit_fields(user, context),
            action=action, status=AuditStatus.ERROR,
            duration_ms=int((time.time() - start_time) * 1000),
            error_message=e.message,
            error_code=e.code,
        )
        return error_response(e.code, e.message)
    except asyncio.CancelledError:
        raise
    except Exception:
        await log_audit(
            session, **_branch_audit_fields(user, context),
            action=action, status=AuditStatus.ERROR,
            duration_ms=int((time.time() - start_time) * 1000),
            error_message="Internal branch operation error",
            error_code="INTERNAL_ERROR",
        )
        return error_response(
            "INTERNAL_ERROR", "Internal branch operation error"
        )

    if log_success:
        await log_audit(
            session, **_branch_audit_fields(user, context),
            action=action, status=AuditStatus.SUCCESS,
            duration_ms=int((time.time() - start_time) * 1000),
            row_count=result_data.get("row_count", 0),
        )
    return result_data, context


async def handle_list_branches(
    user: SQLActor,
    session: AsyncSession,
    *,
    instance_id: str | None = None,
    session_factory=None,
    background_tasks=None,
) -> dict:
    disabled_error = _disabled_user_error(user)
    if disabled_error is not None:
        return disabled_error

    start_time = time.time()
    result = await _execute_branch_sql(
        user, session, sql="SHOW BRANCHES", action="list_branches",
        instance_id=instance_id, session_factory=session_factory,
        background_tasks=background_tasks, max_rows=_BRANCH_LIST_MAX_ROWS,
        log_success=False, read_only=True,
    )
    if isinstance(result, dict):
        return result
    result_data, context = result

    columns = result_data.get("columns", [])
    rows = result_data.get("rows", [])
    branch_index = _branch_column_index(columns)
    if branch_index is None:
        message = "SHOW BRANCHES result does not include a Branch column."
        await log_audit(
            session, **_branch_audit_fields(user, context),
            action="list_branches", status=AuditStatus.ERROR,
            duration_ms=int((time.time() - start_time) * 1000),
            error_message=message,
            error_code="UNEXPECTED_RESULT",
        )
        return error_response("UNEXPECTED_RESULT", message)
    branches = []
    for row in rows:
        if len(row) <= branch_index:
            message = "SHOW BRANCHES result row is missing the Branch column."
            await log_audit(
                session, **_branch_audit_fields(user, context),
                action="list_branches", status=AuditStatus.ERROR,
                duration_ms=int((time.time() - start_time) * 1000),
                error_message=message,
                error_code="UNEXPECTED_RESULT",
            )
            return error_response("UNEXPECTED_RESULT", message)
        branches.append({"branch_name": row[branch_index]})
    await log_audit(
        session, **_branch_audit_fields(user, context),
        action="list_branches", status=AuditStatus.SUCCESS,
        duration_ms=int((time.time() - start_time) * 1000),
        row_count=len(branches),
    )
    payload: dict = {"branches": branches}
    if result_data.get("truncated"):
        payload["truncated"] = True
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


async def handle_create_branch(
    user: SQLActor,
    session: AsyncSession,
    *,
    branch_name: str,
    include_databases: list[str] | None = None,
    instance_id: str | None = None,
    session_factory=None,
    background_tasks=None,
) -> dict:
    disabled_error = _disabled_user_error(user)
    if disabled_error is not None:
        return disabled_error

    valid_branch = validate_identifier_minimal(branch_name, "branch_name")
    if isinstance(valid_branch, dict):
        return valid_branch
    branch_name = valid_branch

    if include_databases is None:
        include_databases = []
    elif not isinstance(include_databases, list):
        return error_response("INVALID_IDENTIFIER", "include_databases must be a list.")

    validated_databases = []
    for database in include_databases:
        valid_database = validate_identifier_minimal(database, "include_databases")
        if isinstance(valid_database, dict):
            return valid_database
        validated_databases.append(valid_database)

    sql = f"CREATE BRANCH {branch_name}"
    if validated_databases:
        sql += " WITH DATABASE " + ", ".join(validated_databases)

    result = await _execute_branch_sql(
        user, session, sql=sql, action="create_branch",
        instance_id=instance_id, session_factory=session_factory,
        background_tasks=background_tasks,
    )
    if isinstance(result, dict):
        payload = _error_payload(result)
        if payload and payload.get("error") == "TIMEOUT":
            return _branch_timeout_unknown("CREATE BRANCH", branch_name)
        return result

    payload = {"branch_name": branch_name, "status": "created"}
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


async def handle_delete_branch(
    user: SQLActor,
    session: AsyncSession,
    *,
    branch_name: str,
    instance_id: str | None = None,
    session_factory=None,
    background_tasks=None,
) -> dict:
    disabled_error = _disabled_user_error(user)
    if disabled_error is not None:
        return disabled_error

    valid_branch = validate_identifier_minimal(branch_name, "branch_name")
    if isinstance(valid_branch, dict):
        return valid_branch
    branch_name = valid_branch

    result = await _execute_branch_sql(
        user, session, sql=f"DROP BRANCH {branch_name}", action="delete_branch",
        delete_branch_name=branch_name,
        instance_id=instance_id, session_factory=session_factory,
        background_tasks=background_tasks,
    )
    if isinstance(result, dict):
        payload = _error_payload(result)
        if payload and payload.get("error") == "TIMEOUT":
            return _branch_timeout_unknown("DROP BRANCH", branch_name)
        return result

    payload = {"branch_name": branch_name, "status": result[0].get("status", "deleted")}
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}
