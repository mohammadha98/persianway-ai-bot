"""Login compatibility tests; no running MongoDB or real accounts required."""
import asyncio
import os
import re
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import bcrypt
import pytest
from bson import ObjectId
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.schemas.user import LoginRequest
from app.services.user_service import UserService

PASSWORD = 'OriginalPassword1!'
HASH = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt(rounds=4)).decode()


def _user(username, email, **overrides):
    now = datetime.now(timezone.utc)
    doc = dict(
        _id=ObjectId(), username=username, email=email, password_hash=HASH,
        role='user', is_active=True, permissions=[], created_at=now,
        updated_at=now, last_login=None,
    )
    doc.update(overrides)
    return doc


class _Cursor:
    def __init__(self, documents):
        self.documents = documents

    def limit(self, count):
        assert count == 2
        self.documents = self.documents[:count]
        return self

    async def to_list(self, length):
        assert length == 2
        return self.documents[:length]


class _Collection:
    def __init__(self, documents):
        self.documents = documents
        self.fallback_queries = []
        self.update_one = AsyncMock()

    async def find_one(self, query):
        return next((doc for doc in self.documents if any(
            doc[field] == value
            for clause in query['$or'] for field, value in clause.items()
        )), None)

    def find(self, query):
        self.fallback_queries.append(query)
        matches = []
        for doc in self.documents:
            for clause in query['$or']:
                field, expression = next(iter(clause.items()))
                assert expression['$options'] == 'i'
                pattern = expression['$regex']
                assert pattern.startswith(r'\A') and pattern.endswith(r'\z')
                # MongoDB PCRE uses \z; Python 3.12 uses \Z for strict end.
                if re.search(pattern[:-2] + r'\Z', doc[field], re.IGNORECASE):
                    matches.append(doc)
                    break
        return _Cursor(matches)


def _service(*documents):
    service = UserService()
    service.secret_key = 'test-only-secret-key-at-least-32-bytes'
    service._collection = _Collection(documents)
    return service


def _login(service, identifier, password=PASSWORD):
    return asyncio.run(service.authenticate_user(
        LoginRequest(username=identifier, password=password)
    ))


@pytest.mark.parametrize('identifier', ['MixedUser', 'mixeduser', 'MIXEDUSER', 'mIxEdUsEr',
                                         'Owner@example.com', 'OWNER@EXAMPLE.COM'])
def test_login_accepts_case_variants_and_preserves_identity(identifier):
    doc = _user('MixedUser', 'Owner@example.com')
    service = _service(doc)
    response = _login(service, identifier)
    assert response.user.id == str(doc['_id'])
    assert response.user.username == 'MixedUser'
    payload = service._decode_access_token(response.access_token)
    assert payload['sub'] == str(doc['_id'])
    assert payload['username'] == 'MixedUser'
    service._collection.update_one.assert_awaited_once()
    assert doc['username'] == 'MixedUser' and doc['password_hash'] == HASH


@pytest.mark.parametrize('identifier', ['Ali', 'ali'])
def test_existing_case_distinct_accounts_keep_exact_login(identifier):
    users = [_user('Ali', 'first@example.com'), _user('ali', 'second@example.com')]
    service = _service(*users)
    response = _login(service, identifier)
    assert response.user.id == str(next(doc['_id'] for doc in users if doc['username'] == identifier))
    assert service._collection.fallback_queries == []


@pytest.mark.parametrize('users,identifier', [
    ([_user('Ali', 'first@example.com'), _user('ali', 'second@example.com')], 'ALI'),
    ([_user('owner@example.com', 'first@example.com'), _user('other', 'Owner@example.com')], 'OWNER@EXAMPLE.COM'),
])
def test_ambiguous_fallback_is_rejected(users, identifier):
    service = _service(*users)
    with pytest.raises(HTTPException) as exc:
        _login(service, identifier)
    assert exc.value.status_code == 401
    assert exc.value.detail == 'Invalid username or password'
    service._collection.update_one.assert_not_awaited()


@pytest.mark.parametrize('identifier', ['.*', 'mixed', 'MixedUser\n', 'missing'])
def test_fallback_is_literal_and_requires_full_identifier(identifier):
    service = _service(_user('MixedUser', 'owner@example.com'))
    with pytest.raises(HTTPException) as exc:
        _login(service, identifier)
    assert exc.value.status_code == 401
    service._collection.update_one.assert_not_awaited()


def test_regex_metacharacters_in_existing_username_are_literal():
    service = _service(_user('User.+[1]', 'owner@example.com'))
    assert _login(service, 'USER.+[1]').user.username == 'User.+[1]'


@pytest.mark.parametrize('identifier', ['MixedUser', 'MIXEDUSER'])
def test_password_remains_case_sensitive(identifier):
    service = _service(_user('MixedUser', 'owner@example.com'))
    with pytest.raises(HTTPException) as exc:
        _login(service, identifier, PASSWORD.lower())
    assert exc.value.status_code == 401
    service._collection.update_one.assert_not_awaited()


def test_disabled_user_cannot_login_via_fallback():
    service = _service(_user('MixedUser', 'owner@example.com', is_active=False))
    with pytest.raises(HTTPException) as exc:
        _login(service, 'MIXEDUSER')
    assert exc.value.status_code == 401
    assert exc.value.detail == 'Account is disabled'
    service._collection.update_one.assert_not_awaited()



def test_wrong_password_for_exact_account_does_not_try_case_variant():
    other_hash = bcrypt.hashpw(b'OtherPassword1!', bcrypt.gensalt(rounds=4)).decode()
    service = _service(
        _user('Ali', 'first@example.com', password_hash=other_hash),
        _user('ali', 'second@example.com'),
    )
    with pytest.raises(HTTPException) as exc:
        _login(service, 'Ali')
    assert exc.value.status_code == 401
    assert service._collection.fallback_queries == []
    service._collection.update_one.assert_not_awaited()
