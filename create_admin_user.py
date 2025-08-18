#!/usr/bin/env python3
"""
Script to create an initial admin user for the Persian Way AI Bot system.
Run this script after setting up the database to create the first admin user.
"""

import asyncio
import sys
import os
from getpass import getpass

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.user_service import UserService
from app.schemas.user import UserCreate, UserRole, PermissionType


async def create_admin_user():
    """Create an initial admin user."""
    print("Creating initial admin user for Persian Way AI Bot")
    print("=" * 50)
    
    # Get user input
    username = input("Enter admin username: ").strip()
    if not username:
        print("Username cannot be empty!")
        return
    
    email = input("Enter admin email: ").strip()
    if not email:
        print("Email cannot be empty!")
        return
    
    full_name = input("Enter admin full name: ").strip()
    if not full_name:
        print("Full name cannot be empty!")
        return
    
    password = getpass("Enter admin password: ")
    if not password:
        print("Password cannot be empty!")
        return
    
    confirm_password = getpass("Confirm admin password: ")
    if password != confirm_password:
        print("Passwords do not match!")
        return
    
    try:
        # Create user service
        user_service = UserService()
        
        # Check if user already exists
        existing_user = await user_service.get_user_by_username(username)
        if existing_user:
            print(f"User with username '{username}' already exists!")
            return
        
        existing_user = await user_service.get_user_by_email(email)
        if existing_user:
            print(f"User with email '{email}' already exists!")
            return
        
        # Create admin user with all permissions
        all_permissions = [
            PermissionType.RAG,
            PermissionType.LLM,
            PermissionType.CONVERSATIONS,
            PermissionType.CONTRIBUTE,
            PermissionType.CHAT,
            PermissionType.GUIDE,
            PermissionType.SETTINGS,
            PermissionType.DOCS
        ]
        
        user_data = UserCreate(
            username=username,
            email=email,
            full_name=full_name,
            password=password,
            role=UserRole.ADMIN,
            is_active=True,
            permissions=all_permissions
        )
        
        # Create the user
        created_user = await user_service.create_user(user_data, created_by="system")
        
        print("\n✅ Admin user created successfully!")
        print(f"Username: {created_user.username}")
        print(f"Email: {created_user.email}")
        print(f"Full Name: {created_user.full_name}")
        print(f"Role: {created_user.role}")
        print(f"Permissions: {[perm.permission_type for perm in created_user.permissions]}")
        print(f"Created At: {created_user.created_at}")
        
    except Exception as e:
        print(f"❌ Error creating admin user: {str(e)}")
        return


if __name__ == "__main__":
    try:
        asyncio.run(create_admin_user())
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")