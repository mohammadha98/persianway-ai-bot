"""Bootstrap (or repair) an admin user for the AI panel at ``/login``.

The panel signs in through ``POST /api/users/login``
(``app/api/routes/users.py`` -> ``UserService.authenticate_user``), which looks
the account up in ``<MONGODB_DATABASE>.users`` by ``username`` *or* ``email``
and verifies ``password_hash`` with bcrypt. When that collection is empty there
is no way in through the API at all, because ``POST /api/users/`` is itself
guarded by ``get_admin_user`` -- the first admin therefore has to be written
out of band.

This script does that through the application's own service layer
(``UserService.create_user`` / ``update_user`` / ``update_user_permissions``),
so the stored document has exactly the shape the API and the panel expect:
``role``/``is_active`` as enum values, ``permissions[]`` entries of
``{permission_type, granted, granted_at, granted_by}``, ``created_at`` /
``updated_at`` / ``last_login`` timestamps and a ``$2b$`` bcrypt hash produced
by ``UserService._hash_password`` (never a hand-rolled hash).

Usage (project root, project venv, WRITES to MongoDB):
    venv/bin/python scripts/create_admin_user.py --username admin \
        --email admin@persianway.co --full-name "Admin"

    # re-apply role/permissions and rotate the password of an existing user
    venv/bin/python scripts/create_admin_user.py --username admin \
        --password 'S3cret-here!' --reset-password

The password is generated with ``secrets`` and printed once when ``--password``
is omitted. Exit codes: 0 = created/updated and verified, 1 = verification
failed, 2 = user already exists and ``--reset-password`` was not given.

Afterwards the login can be confirmed end to end with:
    curl -s -X POST http://127.0.0.1:8000/api/users/login \
        -H 'Content-Type: application/json' \
        -d '{"username":"admin","password":"<the password>"}'
"""
import argparse
import asyncio
import json
import os
import secrets
import string
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import bcrypt  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.schemas.user import (  # noqa: E402
    PermissionType,
    UserCreate,
    UserPermissionUpdate,
    UserRole,
    UserUpdate,
)
from app.services.database import close_database_connection, get_database_service  # noqa: E402
from app.services.user_service import UserService  # noqa: E402

PASSWORD_ALPHABET = string.ascii_letters + string.digits + "!@#$%^&*"
SYMBOLS = "!@#$%^&*"


def generate_password(length: int = 16) -> str:
    """Generate a password that satisfies: lower + upper + digit + symbol."""
    while True:
        candidate = "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(length))
        if (
            any(c.islower() for c in candidate)
            and any(c.isupper() for c in candidate)
            and any(c.isdigit() for c in candidate)
            and any(c in SYMBOLS for c in candidate)
        ):
            return candidate


def role_value(role) -> str:
    """``UserRole`` is a str-enum; accept both the enum and a plain string."""
    return getattr(role, "value", role)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Create or repair an admin user in the panel's users collection."
    )
    parser.add_argument("--username", default="admin", help="login username (default: admin)")
    parser.add_argument("--email", default="admin@persianway.co", help="login email (also usable as username)")
    parser.add_argument("--full-name", default="Admin", help="display name")
    parser.add_argument("--password", default=None, help="password; omit to auto-generate")
    parser.add_argument(
        "--role",
        default=UserRole.ADMIN.value,
        choices=[role.value for role in UserRole],
        help="role to grant (default: admin)",
    )
    parser.add_argument(
        "--no-permissions",
        action="store_true",
        help="create the account without granting any panel permission (default: grant all)",
    )
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="if the user already exists, rotate its password and re-apply role/permissions",
    )
    return parser.parse_args(argv)


async def run(args) -> int:
    password = args.password or generate_password()
    generated = args.password is None
    permissions = [] if args.no_permissions else list(PermissionType)

    db_service = await get_database_service()
    db = db_service.get_database()
    users = db["users"]
    print(f"mongodb uri    : {settings.MONGODB_URL}")
    print(f"database       : {db.name}")
    print(f"users (before) : {await users.count_documents({})}")

    service = UserService()
    existing = await service.get_user_by_username(args.username)

    if existing is not None:
        print(f"\nuser '{args.username}' already exists (id={existing.id}, role={role_value(existing.role)})")
        if not args.reset_password:
            print("nothing to do -- pass --reset-password to rotate the password / re-apply role+permissions")
            return 2
        await service.update_user(
            existing.id,
            UserUpdate(
                email=args.email,
                full_name=args.full_name,
                role=UserRole(args.role),
                is_active=True,
                password=password,
            ),
        )
        if permissions:
            await service.update_user_permissions(
                existing.id,
                UserPermissionUpdate(permissions=permissions, granted_by="system"),
            )
        action = "updated"
    else:
        await service.create_user(
            UserCreate(
                username=args.username,
                email=args.email,
                full_name=args.full_name,
                password=password,
                role=UserRole(args.role),
                is_active=True,
                permissions=permissions,
            ),
            created_by="system",
        )
        action = "created"

    # Independent verification: re-read the raw document and check the hash the
    # same way ``UserService._verify_password`` does (bcrypt.checkpw).
    doc = await users.find_one({"username": args.username})
    if doc is None:
        print(f"\n[FAIL] no document found for username '{args.username}'")
        return 1
    verifies = bcrypt.checkpw(password.encode("utf-8"), doc["password_hash"].encode("utf-8"))
    doc_permissions = [role_value(p.get("permission_type")) for p in doc.get("permissions", [])]

    print(f"\nuser '{args.username}' {action}:")
    print(json.dumps(
        {
            "id": str(doc["_id"]),
            "username": doc["username"],
            "email": doc["email"],
            "full_name": doc.get("full_name"),
            "role": role_value(doc.get("role")),
            "is_active": doc.get("is_active"),
            "permissions": doc_permissions,
            "password_hash_prefix": str(doc["password_hash"])[:4],
            "password_verifies": verifies,
            "created_at": str(doc.get("created_at")),
            "updated_at": str(doc.get("updated_at")),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print("users (after)  :", await users.count_documents({}))

    if generated:
        print(f"\nGENERATED PASSWORD (printed once, change it after the first login): {password}")
    else:
        print("\npassword       : <as provided on the command line>")

    if not verifies:
        print("[FAIL] stored hash does not verify against the supplied password")
        return 1
    print(f"login URL      : http://91.107.148.51/login  (username or email: {args.username})")
    return 0


async def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return await run(args)
    finally:
        await close_database_connection()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
