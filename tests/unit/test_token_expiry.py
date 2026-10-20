"""Access-token lifetime and expiry handling; no MongoDB or running API needed."""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import bcrypt
import jwt
import pytest
from bson import ObjectId
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.core.config import settings
from app.schemas.user import LoginRequest
from app.services.user_service import UserService

THREE_DAYS = timedelta(days=3)
ONE_MINUTE = timedelta(minutes=1)


def _service():
    service = UserService()
    service.secret_key = 'test-only-secret-key-at-least-32-bytes'
    return service


def _remaining_lifetime(token, service):
    """Seconds between now and the token's `exp` claim."""
    payload = jwt.decode(token, service.secret_key, algorithms=[service.algorithm])
    return timedelta(seconds=payload['exp'] - int(datetime.now(timezone.utc).timestamp()))


def test_the_configured_lifetime_is_three_days():
    """The panel used to sign operators out after 30 hard-coded minutes."""
    assert settings.access_token_expire_minutes == 3 * 24 * 60
    assert _service().access_token_expire_minutes == 3 * 24 * 60


def test_minted_tokens_expire_three_days_from_now():
    service = _service()
    token = service._create_access_token({'sub': 'user-1'})
    assert THREE_DAYS - ONE_MINUTE <= _remaining_lifetime(token, service) <= THREE_DAYS


def test_login_reports_the_same_lifetime_to_the_client():
    """`expires_in` drives the panel's fallback expiry, so it must agree."""
    service = _service()
    now = datetime.now(timezone.utc)
    document = {
        '_id': ObjectId(),
        'username': 'operator',
        'email': 'operator@example.com',
        'password_hash': bcrypt.hashpw(b'Secret123!', bcrypt.gensalt(rounds=4)).decode(),
        'role': 'user',
        'is_active': True,
        'permissions': [],
        'created_at': now,
        'updated_at': now,
        'last_login': None,
    }
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=document)
    collection.update_one = AsyncMock()
    service._collection = collection

    response = asyncio.run(
        service.authenticate_user(LoginRequest(username='operator', password='Secret123!'))
    )

    assert response.expires_in == 3 * 24 * 60 * 60
    assert THREE_DAYS - ONE_MINUTE <= _remaining_lifetime(response.access_token, service) <= THREE_DAYS


def test_an_expired_token_is_a_401():
    """Expiry must be a clean 401 so the panel can sign the operator out."""
    service = _service()
    expired = jwt.encode(
        {'sub': 'user-1', 'exp': datetime.now(timezone.utc) - ONE_MINUTE},
        service.secret_key,
        algorithm=service.algorithm,
    )

    with pytest.raises(HTTPException) as raised:
        service._decode_access_token(expired)

    assert raised.value.status_code == 401
    assert raised.value.detail == 'Token has expired'


def test_a_garbage_token_is_a_401():
    service = _service()
    with pytest.raises(HTTPException) as raised:
        service._decode_access_token('not-a-jwt')
    assert raised.value.status_code == 401
