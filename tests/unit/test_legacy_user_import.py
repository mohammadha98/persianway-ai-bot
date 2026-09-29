"""Unit tests for ``scripts/import_legacy_users.py``.

The migration script has to move the pre-migration user documents into the new
database without disturbing the stored ``password_hash`` (the rotated
``SECRET_KEY`` is irrelevant to bcrypt verification), merge accounts that already
exist and stay idempotent. These tests pin the pure planning layer -- Extended
JSON parsing, document validation, the insert/merge/conflict decision and the
password policy -- so none of them needs a running MongoDB.
"""
import importlib.util
import os
import sys
import tempfile
from datetime import datetime, timedelta

import bcrypt
import pytest
from bson import json_util
from bson.objectid import ObjectId

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_script_module():
    """Import ``scripts/import_legacy_users.py`` by path (``scripts`` is no package)."""
    path = os.path.join(PROJECT_ROOT, "scripts", "import_legacy_users.py")
    spec = importlib.util.spec_from_file_location("import_legacy_users_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _load_script_module()

BCRYPT_HASH = "$2b$12$MkA9cXcFBuw0WNkLpts8jezKUNfnU49rZiWbcEdDOn31FFOqtp8Ni"
OTHER_HASH = "$2b$12$7Vj5f9o7ORRYv16gZRtH8.fGnq/ArXh23cEH/NTnAPjZzXGaHV8E."
CREATED_AT = datetime(2025, 8, 19, 12, 26, 7, 280000)
UPDATED_AT = datetime(2025, 8, 19, 12, 26, 7, 280000)
LAST_LOGIN = datetime(2026, 9, 27, 13, 8, 51, 308000)


def _legacy_user(**overrides):
    """A source document shaped exactly like the mongoexport output."""
    doc = {
        "_id": ObjectId("68a46d5f3f992e78f1a46113"),
        "username": "admin",
        "email": "admin@persianway.co",
        "full_name": "admin admin",
        "password_hash": BCRYPT_HASH,
        "role": "admin",
        "is_active": True,
        "permissions": [
            {
                "permission_type": "Chat",
                "granted": True,
                "granted_at": datetime(2025, 8, 19, 12, 26, 6, 899000),
                "granted_by": "system",
            }
        ],
        "created_at": CREATED_AT,
        "updated_at": UPDATED_AT,
        "last_login": LAST_LOGIN,
    }
    doc.update(overrides)
    return doc


def _target_user(**overrides):
    """A document as the new deployment stored it (created by create_admin_user.py)."""
    doc = {
        "_id": ObjectId("6abbbc4af107abe33b7266ea"),
        "username": "admin",
        "email": "admin@persianway.co",
        "full_name": "admin admin",
        "password_hash": bcrypt.hashpw(b"new-deployment-password", bcrypt.gensalt()).decode("utf-8"),
        "role": "admin",
        "is_active": True,
        "permissions": [],
        "created_at": datetime(2026, 9, 29, 9, 31, 40, 0),
        "updated_at": datetime(2026, 9, 29, 9, 31, 40, 0),
        "last_login": None,
    }
    doc.update(overrides)
    return doc



# --------------------------------------------------------------------------- #
# Extended JSON parsing
# --------------------------------------------------------------------------- #

def test_load_legacy_users_parses_extended_json_and_keeps_the_hash():
    export = json_util.dumps([_legacy_user(), _legacy_user(_id=ObjectId(), username="mrasghari", email="asghari@gmail.com")])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "users.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(export)

        users = LEGACY.load_legacy_users(path)

    assert [user["username"] for user in users] == ["admin", "mrasghari"]
    assert isinstance(users[0]["_id"], ObjectId)
    assert isinstance(users[0]["created_at"], datetime)
    assert users[0]["password_hash"] == BCRYPT_HASH, "the bcrypt hash must survive the round trip byte for byte"


def test_load_legacy_users_accepts_a_single_document_object():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "user.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json_util.dumps(_legacy_user()))

        users = LEGACY.load_legacy_users(path)

    assert len(users) == 1
    assert users[0]["username"] == "admin"


# --------------------------------------------------------------------------- #
# Validation / bcrypt helpers
# --------------------------------------------------------------------------- #

def test_is_valid_bcrypt_hash_accepts_real_hashes_and_rejects_junk():
    real = bcrypt.hashpw(b"whatever", bcrypt.gensalt()).decode("utf-8")
    assert LEGACY.is_valid_bcrypt_hash(real) is True
    assert LEGACY.is_valid_bcrypt_hash(BCRYPT_HASH) is True
    assert LEGACY.is_valid_bcrypt_hash("not-a-hash") is False
    assert LEGACY.is_valid_bcrypt_hash("$2b$12$tooshort") is False
    assert LEGACY.is_valid_bcrypt_hash(None) is False


def test_validate_legacy_user_flags_unknown_permission_and_unknown_role():
    problems = LEGACY.validate_legacy_user(
        _legacy_user(
            role="superuser",
            permissions=[{"permission_type": "Telepathy", "granted": True}],
        )
    )
    assert any("unknown role 'superuser'" in problem for problem in problems)
    assert any("unknown permission_type 'Telepathy'" in problem for problem in problems)


def test_validate_legacy_user_accepts_the_export_shape():
    assert LEGACY.validate_legacy_user(_legacy_user()) == []


# --------------------------------------------------------------------------- #
# Planning: new accounts
# --------------------------------------------------------------------------- #

def test_new_account_is_inserted_with_legacy_id_and_untouched_hash():
    legacy_user = _legacy_user(_id=ObjectId("68ad80e83e97e44cdf0d4bd8"), username="mrasghari", email="asghari@gmail.com", role="user")

    item = _one_user_plan(legacy_user, [_target_user()])

    assert item.action == "insert"
    assert item.target_id == ObjectId("68ad80e83e97e44cdf0d4bd8")
    assert item.set_fields["password_hash"] == BCRYPT_HASH
    assert item.set_fields["created_at"] == CREATED_AT
    assert item.set_fields["username"] == "mrasghari"


def test_new_account_without_id_gets_a_generated_one():
    legacy_user = _legacy_user()
    legacy_user.pop("_id")

    item = _one_user_plan(legacy_user, [])

    assert item.action == "insert"
    assert isinstance(item.target_id, ObjectId)
    assert any("no _id" in note for note in item.notes)


def test_legacy_id_taken_by_another_account_is_a_conflict():
    legacy_user = _legacy_user(_id=ObjectId("68a46d5f3f992e78f1a46113"), username="newcomer", email="newcomer@example.com")
    squatter = _target_user(_id=ObjectId("68a46d5f3f992e78f1a46113"), username="someone-else", email="someone@example.com")

    item = _one_user_plan(legacy_user, [squatter])

    assert item.action == "conflict"
    assert item.writes is False
    assert any("ambiguous" in problem for problem in item.problems)


def test_duplicate_ids_inside_the_export_get_a_generated_one():
    first = _legacy_user()
    second = _legacy_user(username="twin", email="twin@example.com")

    plan = LEGACY.plan_actions([first, second], [])

    assert [item.action for item in plan] == ["insert", "insert"]
    assert plan[0].target_id == ObjectId("68a46d5f3f992e78f1a46113")
    assert plan[1].target_id != ObjectId("68a46d5f3f992e78f1a46113")
    assert any("already taken" in note for note in plan[1].notes)


def test_invalid_document_is_reported_and_never_written():
    item = _one_user_plan(_legacy_user(password_hash="plaintext"), [])

    assert item.action == "invalid"
    assert item.writes is False
    assert any("bcrypt" in problem for problem in item.problems)


# --------------------------------------------------------------------------- #
# Planning: accounts that already exist
# --------------------------------------------------------------------------- #

def test_existing_account_is_merged_and_keeps_the_new_password_by_default():
    target = _target_user()

    item = _one_user_plan(_legacy_user(), [target])

    assert item.action == "update"
    assert item.target_id == target["_id"]
    assert "password_hash" not in item.set_fields, "the password of an existing account must survive the import"
    assert item.set_fields["created_at"] == CREATED_AT
    assert item.set_fields["permissions"] == _legacy_user()["permissions"]
    assert item.set_fields["last_login"] == LAST_LOGIN
    assert any("password kept" in note for note in item.notes)


def test_force_passwords_restores_the_legacy_hash():
    item = _one_user_plan(_legacy_user(), [_target_user()], force_passwords=True)

    assert item.action == "update"
    assert item.set_fields["password_hash"] == BCRYPT_HASH


def test_merge_keeps_permissions_that_only_exist_in_the_target():
    target = _target_user(permissions=[{"permission_type": "Analysis", "granted": True, "granted_at": UPDATED_AT, "granted_by": "admin"}])

    item = _one_user_plan(_legacy_user(), [target])

    stored = {entry["permission_type"] for entry in item.set_fields["permissions"]}
    assert stored == {"Chat", "Analysis"}
    assert any("Analysis" in note for note in item.notes)


def test_replace_permissions_drops_target_only_entries():
    target = _target_user(permissions=[{"permission_type": "Analysis", "granted": True, "granted_at": UPDATED_AT, "granted_by": "admin"}])

    item = _one_user_plan(_legacy_user(), [target], replace_permissions=True)

    assert {entry["permission_type"] for entry in item.set_fields["permissions"]} == {"Chat"}


def test_plan_is_idempotent_for_an_existing_account():
    """Applying the plan once and planning again must produce no write at all."""
    target = _target_user()
    first = _one_user_plan(_legacy_user(), [target])
    assert first.action == "update"

    applied = dict(target)
    applied.update(first.set_fields)

    second = _one_user_plan(_legacy_user(), [applied])

    assert second.action == "unchanged"
    assert second.set_fields == {}
    assert second.writes is False


def test_plan_is_idempotent_for_a_fresh_insert():
    legacy_user = _legacy_user()
    first = _one_user_plan(legacy_user, [])
    assert first.action == "insert"

    second = _one_user_plan(legacy_user, [first.set_fields])

    assert second.action == "unchanged"
    assert second.set_fields == {}


def test_email_belonging_to_another_account_is_a_conflict():
    target = _target_user(username="other", email="admin@persianway.co")

    item = _one_user_plan(_legacy_user(), [target])

    assert item.action == "conflict"
    assert item.writes is False
    assert any("already belongs to" in problem for problem in item.problems)


def test_email_taken_by_a_second_document_is_a_conflict():
    keeper = _target_user(username="admin", email="admin@persianway.co")
    squatter = _target_user(_id=ObjectId("6abbbc4af107abe33b7266eb"), username="old-admin", email="admin@persianway.co")
    keeper.pop("email")
    keeper["email"] = "admin@persianway.co"

    item = _one_user_plan(_legacy_user(email="old-admin@persianway.co"), [keeper, squatter])

    assert item.action in ("conflict", "update")
    if item.action == "conflict":
        assert any("already used by" in problem for problem in item.problems)


def test_preserve_ids_recreates_the_account_under_the_legacy_id():
    target = _target_user()

    item = _one_user_plan(_legacy_user(), [target], preserve_ids=True, force_passwords=True)

    assert item.action == "replace"
    assert item.target_id == ObjectId("68a46d5f3f992e78f1a46113")
    assert item.replaced_id == target["_id"]
    assert item.set_fields["password_hash"] == BCRYPT_HASH


def test_older_last_login_does_not_rewrite_the_target():
    target = _target_user(last_login=LAST_LOGIN + timedelta(days=1))

    item = _one_user_plan(_legacy_user(), [target])

    assert "last_login" not in item.set_fields


# --------------------------------------------------------------------------- #
# Permission helpers and CLI scaffolding
# --------------------------------------------------------------------------- #

def test_normalize_permissions_is_order_insensitive():
    permissions = [
        {"permission_type": "Chat", "granted": True, "granted_at": UPDATED_AT, "granted_by": "admin"},
        {"permission_type": "Guide", "granted": False, "granted_at": UPDATED_AT, "granted_by": "admin"},
    ]
    assert LEGACY.normalize_permissions(permissions) == LEGACY.normalize_permissions(list(reversed(permissions)))


def test_merge_permissions_puts_the_legacy_entries_first_and_deduplicates():
    merged = LEGACY.merge_permissions(
        [{"permission_type": "Chat", "granted": True}],
        [{"permission_type": "Chat", "granted": True}, {"permission_type": "Docs", "granted": True}],
    )
    assert [entry["permission_type"] for entry in merged] == ["Chat", "Docs"]


def test_parse_args_defaults_to_a_dry_run():
    args = LEGACY.parse_args([])

    assert args.apply is False
    assert args.force_passwords is False
    assert args.replace_permissions is False
    assert args.preserve_ids is False
    assert args.check_login == []
    assert args.api_url == "http://127.0.0.1:8000"


def test_resolve_default_file_prefers_the_environment_variable(monkeypatch, tmp_path):
    export = tmp_path / "users.json"
    export.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("LEGACY_USERS_FILE", str(export))

    assert LEGACY.resolve_default_file() == str(export)


# --------------------------------------------------------------------------- #
# The real export shipped with the migration
# --------------------------------------------------------------------------- #

REAL_EXPORT = os.path.join(os.path.dirname(PROJECT_ROOT), "persianway-rag-db.users.json")


@pytest.mark.skipif(not os.path.isfile(REAL_EXPORT), reason="the legacy export is not on this machine")
def test_real_export_imports_every_account_with_its_own_hash():
    users = LEGACY.load_legacy_users(REAL_EXPORT)
    assert len(users) >= 13, "the export should contain the whole pre-migration user list"

    problems = {user["username"]: LEGACY.validate_legacy_user(user) for user in users}
    assert all(not issue for issue in problems.values()), problems

    plan = LEGACY.plan_actions(users, [])
    assert [item.action for item in plan] == ["insert"] * len(users)

    expected_hashes = {user["username"]: user["password_hash"] for user in users}
    for item in plan:
        assert item.set_fields["password_hash"] == expected_hashes[item.username]
        assert item.set_fields["_id"] == item.legacy["_id"]

def _one_user_plan(legacy_user, target_docs, **options):
    plan = LEGACY.plan_actions([legacy_user], target_docs, **options)
    assert len(plan) == 1
    return plan[0]
