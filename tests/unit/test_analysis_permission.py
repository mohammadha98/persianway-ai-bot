"""Unit tests for the 'Analysis' permission gate.

The chat page exposes the per-message analysis panel and the AI-trainer feedback
(approve/report) only to users holding the 'Analysis' permission; the server
enforces the same rule on `POST /api/chat/feedback` through the
`require_permission` dependency factory. These tests pin that contract:

* a user holding a granted `Analysis` permission passes,
* a user holding other permissions only (e.g. `Chat`) gets a 403,
* an ungranted `Analysis` entry does not count,
* admins bypass the check entirely.

`require_permission` returns an async dependency callable, so it is awaited with
`asyncio.run` to keep the tests independent of asyncio plugin configuration.
"""
import asyncio
import os
import sys
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.api.routes.users import require_permission
from app.schemas.user import PermissionType, UserPermission, UserResponse, UserRole

_NOW = datetime.now(timezone.utc)


def _make_user(role: UserRole, granted=(), denied=()) -> UserResponse:
    """Build a UserResponse with the given granted / denied permissions."""
    permissions = [
        UserPermission(permission_type=permission, granted=True, granted_at=_NOW, granted_by='tester')
        for permission in granted
    ] + [
        UserPermission(permission_type=permission, granted=False, granted_at=_NOW, granted_by='tester')
        for permission in denied
    ]
    return UserResponse(
        id='507f1f77bcf86cd799439011',
        username='tester',
        email='tester@example.com',
        role=role,
        is_active=True,
        permissions=permissions,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _check(role: UserRole, granted=(), denied=()):
    """Run the Analysis dependency against a synthetic user and return the result."""
    checker = require_permission(PermissionType.ANALYSIS)
    return asyncio.run(checker(_make_user(role, granted, denied)))


def test_analysis_permission_is_exposed_by_the_enum():
    assert PermissionType.ANALYSIS.value == 'Analysis'


def test_user_with_granted_analysis_permission_is_allowed():
    user = _check(UserRole.USER, granted=[PermissionType.CHAT, PermissionType.ANALYSIS])
    assert user.username == 'tester'


def test_user_with_only_chat_permission_is_denied_with_403():
    with pytest.raises(HTTPException) as exc_info:
        _check(UserRole.USER, granted=[PermissionType.CHAT])

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Permission 'Analysis' required"


def test_user_without_any_permission_is_denied():
    with pytest.raises(HTTPException) as exc_info:
        _check(UserRole.USER)

    assert exc_info.value.status_code == 403


def test_ungranted_analysis_entry_does_not_count():
    with pytest.raises(HTTPException) as exc_info:
        _check(UserRole.USER, denied=[PermissionType.ANALYSIS])

    assert exc_info.value.status_code == 403


def test_admin_bypasses_the_permission_check():
    admin = _check(UserRole.ADMIN)
    assert admin.role == UserRole.ADMIN
