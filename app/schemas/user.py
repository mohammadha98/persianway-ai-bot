from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class PermissionType(str, Enum):
    """Available permission types for users."""
    RAG = "RAG"
    LLM = "LLM"
    CONVERSATIONS = "Conversations"
    CONTRIBUTE = "Contribute"
    CHAT = "Chat"
    GUIDE = "Guide"
    SETTINGS = "Settings"
    DOCS = "Docs"


class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"


class UserPermission(BaseModel):
    """Schema for user permission."""
    permission_type: PermissionType = Field(..., description="Type of permission")
    granted: bool = Field(default=True, description="Whether the permission is granted")
    granted_at: datetime = Field(default_factory=datetime.utcnow, description="When the permission was granted")
    granted_by: Optional[str] = Field(None, description="User ID who granted this permission")


class UserBase(BaseModel):
    """Base user schema with common fields."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, max_length=100, description="User's full name")
    role: UserRole = Field(default=UserRole.USER, description="User role in the system")
    is_active: bool = Field(default=True, description="Whether the user account is active")


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, description="User password (minimum 8 characters)")
    permissions: Optional[List[PermissionType]] = Field(default=[], description="List of permissions to grant")


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Unique username")
    email: Optional[EmailStr] = Field(None, description="User email address")
    full_name: Optional[str] = Field(None, max_length=100, description="User's full name")
    role: Optional[UserRole] = Field(None, description="User role in the system")
    is_active: Optional[bool] = Field(None, description="Whether the user account is active")
    password: Optional[str] = Field(None, min_length=8, description="New password (minimum 8 characters)")


class UserPermissionUpdate(BaseModel):
    """Schema for updating user permissions."""
    permissions: List[PermissionType] = Field(..., description="List of permissions to grant")
    granted_by: str = Field(..., description="User ID who is granting these permissions")


class UserResponse(UserBase):
    """Schema for user response data."""
    id: str = Field(..., description="User ID")
    permissions: List[UserPermission] = Field(default=[], description="User permissions")
    created_at: datetime = Field(..., description="When the user was created")
    updated_at: datetime = Field(..., description="When the user was last updated")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    class Config:
        from_attributes = True


class UserDocument(BaseModel):
    """Schema for user document in database."""
    username: str
    email: str
    full_name: Optional[str] = None
    password_hash: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    permissions: List[UserPermission] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None


class UserListResponse(BaseModel):
    """Schema for paginated user list response."""
    users: List[UserResponse] = Field(..., description="List of users")
    total_count: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of users per page")
    has_next: bool = Field(..., description="Whether there are more pages")


class LoginRequest(BaseModel):
    """Schema for user login request."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    """Schema for login response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserResponse = Field(..., description="User information")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class PasswordChangeRequest(BaseModel):
    """Schema for password change request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (minimum 8 characters)")