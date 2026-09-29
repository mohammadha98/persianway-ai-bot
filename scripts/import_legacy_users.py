"""Restore the pre-migration user accounts into a freshly migrated database.

Why the rotated JWT secret is *not* the problem
-----------------------------------------------
``UserService.authenticate_user`` (``app/services/user_service.py``) looks the
account up by ``username`` **or** ``email`` and then verifies the stored
``password_hash`` with ``bcrypt.checkpw`` -- the JWT secret never takes part in
that step. ``SECRET_KEY`` only signs/verifies the *issued* tokens. A rotated
secret therefore does not lock anybody out of the new deployment: every user
signs in once more with the password they already have and the fresh token is
signed with the new secret. The only casualty of the rotation is a token issued
by the old deployment, and those expire after 30 minutes anyway.

The migration therefore only has to reproduce the ``users`` documents -- above
all the ``password_hash`` **byte for byte**. This script validates each source
document and then writes it *verbatim*: no re-hashing, no re-normalisation, so
every existing password keeps working.

Input
-----
A ``mongoexport`` / ``mongodump`` style Extended JSON export (``{"$oid": ...}``,
``{"$date": ...}``). By default the file next to the project root
(``../persianway-rag-db.users.json``) is used; ``--file`` or
``LEGACY_USERS_FILE`` override it.

Behaviour
---------
* dry run unless ``--apply`` is given,
* every write is preceded by a JSON backup of the current ``users`` collection,
* idempotent: a second run reports ``unchanged`` and writes nothing,
* accounts are matched by ``username``, then ``email``, then ``_id``; a match is
  *merged* (the legacy values win), never duplicated,
* the password of an account that already exists is kept unless
  ``--force-passwords`` asks for the legacy hash (default: keep, so the admin
  password generated for the new deployment keeps working),
* conflicting unique keys (the same username *or* email belonging to two
  different accounts) are reported and skipped instead of force-written,
* ``--preserve-ids`` also writes the legacy ``_id`` for an account that already
  exists under that username (drops and re-inserts, so use it before anything
  else references the new ``_id``).

Verification (always runs)
--------------------------
* field-by-field re-read of every written document,
* the stored hash must still be a parseable bcrypt hash,
* ``UserService.get_user_by_username`` -- the API's own read path -- must build a
  ``UserResponse``,
* with ``--check-login user=password`` a real ``POST /api/users/login`` is made
  against ``--api-url`` and the returned JWT is decoded with the *current*
  ``settings.secret_key``.

Usage (project root, project venv; only ``--apply`` writes to MongoDB):
    venv/bin/python scripts/import_legacy_users.py                 # dry run
    venv/bin/python scripts/import_legacy_users.py --apply
    venv/bin/python scripts/import_legacy_users.py --apply --replace-permissions
    venv/bin/python scripts/import_legacy_users.py --apply --check-login admin=<pw>

Exit codes: 0 = imported and verified, 1 = verification failed or conflicts were
found, 2 = preconditions failed (missing/unreadable file, empty export).
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import bcrypt  # noqa: E402
import jwt  # noqa: E402
from bson import json_util  # noqa: E402
from bson.objectid import ObjectId  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.schemas.user import PermissionType, UserDocument, UserRole  # noqa: E402
from app.services.database import close_database_connection, get_database_service  # noqa: E402
from app.services.user_service import UserService  # noqa: E402

DEFAULT_FILE_CANDIDATES = (
    os.path.join(os.path.dirname(PROJECT_ROOT), "persianway-rag-db.users.json"),
    os.path.join(PROJECT_ROOT, "persianway-rag-db.users.json"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "persian_way_ai.users.json"),
)

# Fields compared against the source before deciding that a document has to be
# rewritten. ``updated_at`` is deliberately absent (it is set to "now" whenever a
# document is written) and ``last_login`` is merged separately, so that a re-run
# of the script finds nothing to do.
COMPARED_FIELDS = ("username", "email", "full_name", "role", "is_active", "permissions", "created_at")

PERMISSION_TYPES = {item.value for item in PermissionType}
USER_ROLES = {item.value for item in UserRole}

def resolve_default_file() -> Optional[str]:
    """First existing candidate for the legacy export."""
    env_file = os.environ.get("LEGACY_USERS_FILE")
    if env_file:
        return env_file
    for candidate in DEFAULT_FILE_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_legacy_users(path: str) -> List[Dict[str, Any]]:
    """Read an Extended JSON export into real ``ObjectId`` / ``datetime`` values.

    ``json_util`` produces naive UTC datetimes, which is exactly what the
    application stores (``datetime.utcnow()``), so documents round-trip unchanged.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    data = json_util.loads(raw.decode("utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array (or a single object) of user documents")
    return [doc for doc in data if isinstance(doc, dict)]


def role_value(role: Any) -> Optional[str]:
    """``UserRole`` is a str-enum; accept the enum as well as a plain string."""
    value = getattr(role, "value", role)
    return None if value is None else str(value)


def permission_value(permission_type: Any) -> Optional[str]:
    """``PermissionType`` is a str-enum; accept the enum as well as a string."""
    value = getattr(permission_type, "value", permission_type)
    return None if value is None else str(value)


def is_valid_bcrypt_hash(value: Any) -> bool:
    """True when ``value`` is a bcrypt hash this deployment could verify.

    ``bcrypt.checkpw`` rejects malformed input with ``ValueError``; a well formed
    hash simply returns ``False`` for the throw-away probe password.
    """
    if not isinstance(value, str) or not value.startswith(("$2a$", "$2b$", "$2y$")):
        return False
    try:
        bcrypt.checkpw(b"__import_probe__", value.encode("utf-8"))
    except ValueError:
        return False
    return True


def normalize_permissions(permissions: Optional[Iterable[Dict[str, Any]]]) -> List[Tuple[str, bool]]:
    """Sorted ``(permission_type, granted)`` view of a permission list."""
    pairs = []
    for item in permissions or []:
        if not isinstance(item, dict):
            continue
        pairs.append((permission_value(item.get("permission_type")) or "", bool(item.get("granted", True))))
    return sorted(pairs)


def merge_permissions(
    legacy: Optional[Sequence[Dict[str, Any]]],
    current: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Legacy permissions win, permissions that only exist in the target survive.

    The target of a merge is the *new* deployment, so a permission that an admin
    granted there after the migration must not silently disappear.
    """
    merged: List[Dict[str, Any]] = []
    seen: List[Tuple[str, bool]] = []
    for item in list(legacy or []) + list(current or []):
        if not isinstance(item, dict):
            continue
        key = (permission_value(item.get("permission_type")) or "", bool(item.get("granted", True)))
        if key in seen:
            continue
        seen.append(key)
        merged.append(dict(item))
    return merged


def validate_legacy_user(doc: Dict[str, Any]) -> List[str]:
    """Return the schema violations of one source document (empty list = valid).

    Validation is only a guard rail: the document that reaches MongoDB is the
    source document itself, so a valid account is never re-hashed or rewritten.
    """
    problems: List[str] = []
    payload = {key: value for key, value in doc.items() if key != "_id"}

    for required in ("username", "email", "password_hash"):
        if not payload.get(required):
            problems.append(f"missing field '{required}'")

    if payload.get("password_hash") and not is_valid_bcrypt_hash(payload["password_hash"]):
        problems.append("password_hash is not a parseable bcrypt hash")

    role = role_value(payload.get("role"))
    if role is not None and role not in USER_ROLES:
        problems.append(f"unknown role '{role}'")

    for index, permission in enumerate(payload.get("permissions") or []):
        if not isinstance(permission, dict):
            problems.append(f"permissions[{index}] is not an object")
            continue
        permission_type = permission_value(permission.get("permission_type"))
        if permission_type not in PERMISSION_TYPES:
            problems.append(f"permissions[{index}] has unknown permission_type '{permission_type}'")

    try:
        UserDocument(**payload)
    except ValidationError as exc:
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()))
            problems.append(f"{location}: {error.get('msg')}")

    return problems

@dataclass
class PlanItem:
    """One source account and what the script would do with it."""

    action: str  # insert | update | replace | unchanged | conflict | invalid
    username: str
    legacy: Dict[str, Any]
    target_id: Optional[ObjectId] = None
    replaced_id: Optional[ObjectId] = None
    set_fields: Dict[str, Any] = field(default_factory=dict)
    changes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def writes(self) -> bool:
        return self.action in ("insert", "update", "replace")

    @property
    def failed(self) -> bool:
        return self.action in ("conflict", "invalid")


def _values_differ(field_name: str, desired: Any, actual: Any) -> bool:
    if field_name == "permissions":
        return normalize_permissions(desired) != normalize_permissions(actual)
    if field_name == "role":
        return role_value(desired) != role_value(actual)
    return desired != actual


def _describe(field_name: str, desired: Any, actual: Any) -> str:
    if field_name == "permissions":
        return f"permissions: {len(normalize_permissions(actual))} -> {len(normalize_permissions(desired))} entries"
    if field_name == "is_active":
        return f"is_active: {bool(actual)} -> {bool(desired)}"
    return f"{field_name}: {actual!r} -> {desired!r}"

def plan_actions(
    legacy_users: Sequence[Dict[str, Any]],
    existing_docs: Sequence[Dict[str, Any]],
    *,
    force_passwords: bool = False,
    replace_permissions: bool = False,
    preserve_ids: bool = False,
) -> List[PlanItem]:
    """Work out the write set without touching the database.

    Pure function: the same inputs always produce the same plan, which is what
    makes both the dry run and ``tests/unit/test_legacy_user_import.py`` honest.
    """
    by_username = {doc.get("username"): doc for doc in existing_docs}
    by_email = {doc.get("email"): doc for doc in existing_docs}
    by_id = {doc.get("_id"): doc for doc in existing_docs}
    claimed_ids = set(by_id)
    plan: List[PlanItem] = []

    for legacy in legacy_users:
        username = str(legacy.get("username") or "")
        email = legacy.get("email")
        legacy_id = legacy.get("_id")
        item = PlanItem(action="unchanged", username=username, legacy=legacy)

        problems = validate_legacy_user(legacy)
        if problems:
            item.action = "invalid"
            item.problems = problems
            plan.append(item)
            continue

        # Identity is matched by username first, then email, then the legacy _id.
        # A username match is a real identity; the other two are only hints, so a
        # differing username turns them into a conflict further down.
        matched_by = "username"
        target = by_username.get(username)
        if target is None and email is not None:
            matched_by = "email"
            target = by_email.get(email)
        if target is None and legacy_id is not None:
            matched_by = "_id"
            target = by_id.get(legacy_id)

        if target is None:
            # --- brand new account: write the source document as it is ---------
            write_id = legacy_id
            if write_id is None:
                write_id = ObjectId()
                item.notes.append("source has no _id, a new one was generated")
            elif write_id in claimed_ids:
                write_id = ObjectId()
                item.notes.append("legacy _id is already taken by another document, a new one was generated")
            claimed_ids.add(write_id)

            item.action = "insert"
            item.target_id = write_id
            item.set_fields = dict(legacy, _id=write_id)
            item.changes.append(
                "insert with role={role}, permissions={count}, bcrypt hash copied verbatim".format(
                    role=role_value(legacy.get("role")), count=len(legacy.get("permissions") or [])
                )
            )
            plan.append(item)
            continue

        # --- account already present: merge the legacy values in ---------------
        item.target_id = target.get("_id")
        if target.get("username") != username:
            item.action = "conflict"
            if matched_by == "_id":
                item.problems.append(
                    f"legacy _id {legacy_id} is already used by the existing user "
                    f"'{target.get('username')}', so the identity is ambiguous"
                )
            else:
                item.problems.append(
                    f"email '{email}' already belongs to the existing user '{target.get('username')}'"
                )
            plan.append(item)
            continue
        email_owner = by_email.get(email)
        if email_owner is not None and email_owner.get("_id") != target.get("_id"):
            item.action = "conflict"
            item.problems.append(
                f"email '{email}' is already used by the existing user '{email_owner.get('username')}'"
            )
            plan.append(item)
            continue

        if preserve_ids and legacy_id is not None and legacy_id != target.get("_id"):
            if legacy_id in claimed_ids:
                item.notes.append(
                    f"legacy _id {legacy_id} is taken by '{by_id[legacy_id].get('username')}', "
                    f"keeping {target.get('_id')}"
                )
            else:
                claimed_ids.add(legacy_id)
                item.action = "replace"
                item.target_id = legacy_id
                item.replaced_id = target.get("_id")
                item.set_fields = dict(legacy, _id=legacy_id)
                item.changes.append(
                    f"re-create '{username}' under the legacy _id {legacy_id} "
                    f"(currently {target.get('_id')}); the previous document is dropped"
                )
                plan.append(item)
                continue

        if replace_permissions:
            desired_permissions = list(legacy.get("permissions") or [])
        else:
            desired_permissions = merge_permissions(legacy.get("permissions"), target.get("permissions"))
            legacy_keys = normalize_permissions(legacy.get("permissions"))
            extra = sorted(
                permission_value(entry.get("permission_type"))
                for entry in target.get("permissions") or []
                if (permission_value(entry.get("permission_type")), bool(entry.get("granted", True)))
                not in legacy_keys
            )
            if extra:
                item.notes.append("permissions only present in the target are kept: " + ", ".join(extra))

        desired: Dict[str, Any] = {
            "username": username,
            "email": email,
            "full_name": legacy.get("full_name"),
            "role": role_value(legacy.get("role")),
            "is_active": bool(legacy.get("is_active", True)),
            "permissions": desired_permissions,
            "created_at": legacy.get("created_at"),
        }
        if force_passwords:
            desired["password_hash"] = legacy.get("password_hash")

        set_fields: Dict[str, Any] = {}
        for field_name in COMPARED_FIELDS:
            if _values_differ(field_name, desired.get(field_name), target.get(field_name)):
                set_fields[field_name] = desired.get(field_name)
                item.changes.append(_describe(field_name, desired.get(field_name), target.get(field_name)))

        if force_passwords:
            if desired["password_hash"] != target.get("password_hash"):
                set_fields["password_hash"] = desired["password_hash"]
                item.changes.append("password_hash: replaced with the legacy hash (--force-passwords)")
        else:
            item.notes.append("password kept as stored in the target (--force-passwords to use the legacy hash)")

        legacy_last_login = legacy.get("last_login")
        target_last_login = target.get("last_login")
        if legacy_last_login is not None and (target_last_login is None or legacy_last_login > target_last_login):
            set_fields["last_login"] = legacy_last_login
            item.changes.append(f"last_login: {target_last_login!r} -> {legacy_last_login!r}")

        if set_fields:
            item.action = "update"
            set_fields["updated_at"] = datetime.utcnow()
            item.set_fields = set_fields
        else:
            item.action = "unchanged"
            item.notes.append("target already matches the source")
        plan.append(item)

    return plan

def write_backup(documents: Sequence[Dict[str, Any]], backup_dir: str) -> str:
    """Dump the current collection to Extended JSON before anything is written."""
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(backup_dir, f"users-backup-{stamp}.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json_util.dumps(list(documents), indent=2, ensure_ascii=False))
    return path


async def apply_plan(collection, plan: Sequence[PlanItem]) -> Dict[str, int]:
    """Write the plan through the same collection the API uses."""
    from pymongo.errors import DuplicateKeyError

    stats = {"inserted": 0, "updated": 0, "replaced": 0, "unchanged": 0, "conflict": 0, "invalid": 0, "errors": 0}

    for item in plan:
        if item.action == "unchanged":
            stats["unchanged"] += 1
            continue
        if item.failed:
            stats["conflict" if item.action == "conflict" else "invalid"] += 1
            continue
        try:
            if item.action == "insert":
                await collection.insert_one(item.set_fields)
                stats["inserted"] += 1
            elif item.action == "update":
                result = await collection.update_one({"_id": item.target_id}, {"$set": item.set_fields})
                if result.matched_count == 0:
                    raise RuntimeError(f"document {item.target_id} disappeared before the update")
                stats["updated"] += 1
            elif item.action == "replace":
                await collection.delete_one({"_id": item.replaced_id})
                await collection.insert_one(item.set_fields)
                stats["replaced"] += 1
        except DuplicateKeyError as exc:
            item.action = "conflict"
            item.problems.append(f"duplicate key: {exc}")
            stats["errors"] += 1
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            item.problems.append(f"{type(exc).__name__}: {exc}")
            stats["errors"] += 1

    return stats

async def verify_plan(
    collection,
    plan: Sequence[PlanItem],
    service: UserService,
    *,
    force_passwords: bool = False,
    replace_permissions: bool = False,
) -> Tuple[List[str], List[str]]:
    """Re-read every target document and compare it with the source account.

    Returns ``(failures, report_lines)``. The API's own read path
    (``UserService.get_user_by_username`` -> ``_document_to_response``) runs for
    every account as well, so a document that the panel could not render is
    treated as a failed import.
    """
    failures: List[str] = []
    lines: List[str] = []

    for item in plan:
        if item.failed:
            lines.append(f"[{item.action:8}] {item.username:24} NOT IMPORTED: {'; '.join(item.problems)}")
            continue
        if item.target_id is None:
            failures.append(f"{item.username}: no target _id was planned")
            continue

        doc = await collection.find_one({"_id": item.target_id})
        if doc is None:
            failures.append(f"{item.username}: document {item.target_id} is missing after the write")
            lines.append(f"[FAIL    ] {item.username:24} document not found")
            continue

        legacy = item.legacy
        problems: List[str] = []

        if doc.get("username") != legacy.get("username"):
            problems.append(f"username {doc.get('username')!r} != {legacy.get('username')!r}")
        if doc.get("email") != legacy.get("email"):
            problems.append(f"email {doc.get('email')!r} != {legacy.get('email')!r}")
        if role_value(doc.get("role")) != role_value(legacy.get("role")):
            problems.append(f"role {role_value(doc.get('role'))!r} != {role_value(legacy.get('role'))!r}")
        if bool(doc.get("is_active", True)) != bool(legacy.get("is_active", True)):
            problems.append("is_active differs")

        stored_hash = doc.get("password_hash")
        if not is_valid_bcrypt_hash(stored_hash):
            problems.append("stored password_hash is not a parseable bcrypt hash")
        hash_is_legacy = stored_hash == legacy.get("password_hash")
        password_kept = item.action in ("update", "unchanged") and not force_passwords
        if not hash_is_legacy and not password_kept:
            problems.append("stored password_hash is neither the legacy hash nor a deliberately kept target hash")

        legacy_permissions = set(normalize_permissions(legacy.get("permissions")))
        stored_permissions = set(normalize_permissions(doc.get("permissions")))
        if not legacy_permissions.issubset(stored_permissions):
            problems.append(
                "missing permissions: "
                + ", ".join(sorted(f"{name}{'' if granted else '(denied)'}" for name, granted in legacy_permissions - stored_permissions))
            )
        if replace_permissions and legacy_permissions != stored_permissions:
            problems.append("permissions were not replaced exactly as requested")

        legacy_created_at = legacy.get("created_at")
        stored_created_at = doc.get("created_at")
        if legacy_created_at is not None and stored_created_at is not None:
            if abs((stored_created_at - legacy_created_at).total_seconds()) > 0.001:
                problems.append(f"created_at {stored_created_at!r} != {legacy_created_at!r}")

        # The panel reads accounts through the service layer, so exercise it too.
        try:
            response = await service.get_user_by_username(str(legacy.get("username")))
            if response is None:
                problems.append("UserService.get_user_by_username() returned None")
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            problems.append(f"UserService.get_user_by_username() raised {type(exc).__name__}: {exc}")

        password_state = "legacy-hash" if hash_is_legacy else "kept-target-hash"
        if problems:
            failures.append(f"{item.username}: " + "; ".join(problems))
            lines.append(f"[FAIL    ] {item.username:24} {'; '.join(problems)}")
        else:
            lines.append(
                f"[OK      ] {item.username:24} role={role_value(doc.get('role')):5} "
                f"perms={len(stored_permissions):2} password={password_state} active={bool(doc.get('is_active', True))}"
            )

    return failures, lines

def _http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    timeout: float = 15.0,
) -> Tuple[int, Any, str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator supplied URL
            return response.status, json.loads(response.read().decode("utf-8", "replace") or "null"), ""
    except urllib.error.HTTPError as exc:
        return exc.code, None, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return 0, None, f"{type(exc).__name__}: {exc}"


def check_login(api_url: str, username: str, password: str) -> Tuple[bool, List[str], Optional[str]]:
    """Prove the whole chain: password hash -> JWT signed with the *new* secret.

    Returns ``(ok, report_lines, token)``.
    """
    base = api_url.rstrip("/")
    status, payload, raw = _http_json(
        "POST", f"{base}/api/users/login", {"username": username, "password": password}
    )
    token = payload.get("access_token") if isinstance(payload, dict) else None
    if status != 200 or not token:
        return False, [f"POST {base}/api/users/login as '{username}' -> HTTP {status} {raw[:200]}"], None

    lines = [
        f"POST {base}/api/users/login as '{username}' -> 200 "
        f"(token_type={payload.get('token_type')}, expires_in={payload.get('expires_in')})"
    ]
    ok = True
    try:
        claims = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        expires_at = datetime.utcfromtimestamp(claims["exp"]).isoformat() + "Z" if "exp" in claims else "?"
        lines.append(
            "JWT verified against the current settings.secret_key: "
            f"sub={claims.get('sub')} username={claims.get('username')} role={claims.get('role')} exp={expires_at}"
        )
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        ok = False
        lines.append(f"JWT could not be verified with settings.secret_key: {type(exc).__name__}: {exc}")

    me_status, me_payload, me_raw = _http_json("GET", f"{base}/api/users/me", token=token)
    if me_status == 200 and isinstance(me_payload, dict):
        lines.append(
            f"GET {base}/api/users/me -> 200 as '{me_payload.get('username')}' "
            f"(role={role_value(me_payload.get('role'))}, permissions={len(me_payload.get('permissions') or [])})"
        )
    else:
        ok = False
        lines.append(f"GET {base}/api/users/me -> HTTP {me_status} {me_raw[:200]}")

    return ok, lines, token


def render_plan(plan: Sequence[PlanItem]) -> List[str]:
    lines = []
    for item in plan:
        details = "; ".join(item.changes) or "no change"
        if item.notes:
            details += f"  [{'; '.join(item.notes)}]"
        lines.append(f"[{item.action:8}] {item.username:24} {details}")
    return lines

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the pre-migration user accounts (bcrypt hashes included) into the current database."
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Extended JSON export of the old users collection "
        "(default: ../persianway-rag-db.users.json, or $LEGACY_USERS_FILE)",
    )
    parser.add_argument("--apply", action="store_true", help="write to MongoDB (default: dry run)")
    parser.add_argument("--database", default=None, help="override settings.MONGODB_DATABASE (useful for a rehearsal)")
    parser.add_argument(
        "--force-passwords",
        action="store_true",
        help="also overwrite the password_hash of accounts that already exist in the target",
    )
    parser.add_argument(
        "--replace-permissions",
        action="store_true",
        help="store exactly the legacy permission list instead of merging it with the target's",
    )
    parser.add_argument(
        "--preserve-ids",
        action="store_true",
        help="re-create accounts that already exist under their legacy _id (drops the target document)",
    )
    parser.add_argument("--no-backup", action="store_true", help="skip the JSON backup taken before the writes")
    parser.add_argument(
        "--backup-dir",
        default=os.path.join(tempfile.gettempdir(), "persianway_user_backups"),
        help="where the pre-write backup is written (default: the system temp directory)",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="base URL used by --check-login (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--check-login",
        action="append",
        default=[],
        metavar="USER=PASSWORD",
        help="after the import, sign in through POST /api/users/login with these credentials (repeatable)",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    file_path = args.file or resolve_default_file()
    if not file_path or not os.path.isfile(file_path):
        candidates = "\n  ".join(DEFAULT_FILE_CANDIDATES)
        print(f"[FAIL] legacy export not found: {file_path!r}. Pass --file or set LEGACY_USERS_FILE. Looked at:\n  {candidates}")
        return 2

    try:
        legacy_users = load_legacy_users(file_path)
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        print(f"[FAIL] cannot parse {file_path}: {type(exc).__name__}: {exc}")
        return 2

    if not legacy_users:
        print(f"[FAIL] {file_path} contains no user documents")
        return 2

    if args.database and args.database != settings.MONGODB_DATABASE:
        # ``UserService`` builds its collection from ``settings.MONGODB_DATABASE``,
        # so the override has to be applied there too -- otherwise the verification
        # below would check one database while the script wrote into another.
        print(
            f"note           : settings.MONGODB_DATABASE {settings.MONGODB_DATABASE!r} -> {args.database!r} "
            "for this run (the running API still serves the configured one)"
        )
        settings.MONGODB_DATABASE = args.database
        if args.check_login:
            print("note           : --check-login is skipped, it would exercise the live API against the configured database")
            args.check_login = []

    db_service = await get_database_service()
    db = db_service.get_database()
    # ``UserService`` reads the literal "users" collection (app/services/user_service.py).
    collection = db["users"]

    print(f"export file    : {file_path} ({len(legacy_users)} user documents)")
    print(f"target         : {db.name}.users @ {settings.MONGODB_URL}")
    print(f"mode           : {'APPLY (writes to MongoDB)' if args.apply else 'dry run (no writes)'}")
    print(
        "passwords      : "
        + (
            "legacy hash wins, existing hashes overwritten"
            if args.force_passwords
            else "new accounts get the legacy hash, existing accounts keep theirs"
        )
    )
    print(
        "permissions    : "
        + ("replaced by the legacy list" if args.replace_permissions else "merged (legacy wins, target-only entries kept)")
    )
    print(
        "ids            : "
        + (
            "legacy _id preserved and re-applied to existing accounts"
            if args.preserve_ids
            else "legacy _id used for new accounts only"
        )
    )

    existing = await collection.find({}).to_list(length=None)
    print(f"users (before) : {len(existing)}")

    plan = plan_actions(
        legacy_users,
        existing,
        force_passwords=args.force_passwords,
        replace_permissions=args.replace_permissions,
        preserve_ids=args.preserve_ids,
    )

    print("\n--- plan ---")
    for line in render_plan(plan):
        print(line)

    stats: Dict[str, int] = {}
    if args.apply:
        if not args.no_backup:
            print(f"\nbackup         : {write_backup(existing, args.backup_dir)}")
        stats = await apply_plan(collection, plan)
        print(
            "writes         : inserted={inserted} updated={updated} replaced={replaced} "
            "unchanged={unchanged} conflicts={conflict} invalid={invalid} errors={errors}".format(**stats)
        )
    else:
        print("\ndry run: nothing was written yet (re-run with --apply)")

    failures: List[str] = []
    if args.apply:
        failures, verify_lines = await verify_plan(
            collection,
            plan,
            UserService(),
            force_passwords=args.force_passwords,
            replace_permissions=args.replace_permissions,
        )
        print("\n--- verification ---")
        for line in verify_lines:
            print(line)
        print(f"users (after)  : {await collection.count_documents({})}")
    else:
        print("\n--- verification ---")
        print("dry run: nothing was written, so there is nothing to re-read yet (re-run with --apply)")

    login_ok = True
    if args.check_login:
        print("\n--- login check ---")
        for pair in args.check_login:
            username, separator, password = pair.partition("=")
            if not separator:
                print(f"[FAIL    ] --check-login expects USER=PASSWORD, got {pair!r}")
                login_ok = False
                continue
            ok, lines, _ = check_login(args.api_url, username, password)
            login_ok = login_ok and ok
            for line in lines:
                print(f"[{'OK      ' if ok else 'FAIL    '}] {line}")

    conflicts = [item for item in plan if item.failed]
    errors = int(stats.get("errors", 0))
    problem_count = len(failures) + len(conflicts) + errors + (0 if login_ok else 1)

    print("\n--- summary ---")
    print(f"accounts in export : {len(plan)}")
    print(f"verified           : {len(plan) - len(failures) - len(conflicts)}")
    print(f"verification errors: {len(failures)}")
    print(f"conflicts/invalid  : {len(conflicts)}")
    print(f"write errors       : {errors}")
    if args.apply and not args.force_passwords:
        print(
            "note               : accounts that already existed keep their previous password "
            "(scripts/create_admin_user.py --reset-password rotates one)."
        )
    print(
        "note               : the rotated SECRET_KEY does not touch the stored bcrypt hashes -- every user "
        "signs in once more with the password they already have, while tokens minted by the old deployment "
        "are rejected and expire within 30 minutes."
    )
    print(f"login URL          : {args.api_url.rstrip('/')}/login")

    return 1 if problem_count else 0


async def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return await run(args)
    finally:
        await close_database_connection()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
