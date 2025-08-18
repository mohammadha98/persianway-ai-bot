from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from pymongo import DESCENDING
from motor.motor_asyncio import AsyncIOMotorCollection
import bcrypt
import jwt
from fastapi import HTTPException, status

from app.schemas.user import (
    UserDocument,
    UserCreate,
    UserUpdate,
    UserResponse,
    UserListResponse,
    UserPermission,
    PermissionType,
    UserRole,
    LoginRequest,
    LoginResponse,
    PasswordChangeRequest,
    UserPermissionUpdate
)
from app.services.database import get_database_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class UserService:
    """Service for managing users and permissions."""
    
    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None
        self.secret_key = getattr(settings, 'SECRET_KEY', 'your-secret-key-here')
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
    
    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get the users collection."""
        if self._collection is None:
            db_service = await get_database_service()
            db = db_service.get_database()
            self._collection = db.get_collection("users")
            
            # Create indexes
            await self._collection.create_index("username", unique=True)
            await self._collection.create_index("email", unique=True)
            
        return self._collection
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def _create_access_token(self, data: dict) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def _decode_access_token(self, token: str) -> Dict[str, Any]:
        """Decode and verify a JWT access token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def create_user(self, user_data: UserCreate, created_by: Optional[str] = None) -> UserResponse:
        """Create a new user."""
        try:
            collection = await self._get_collection()
            
            # Check if username or email already exists
            existing_user = await collection.find_one({
                "$or": [
                    {"username": user_data.username},
                    {"email": user_data.email}
                ]
            })
            
            if existing_user:
                if existing_user["username"] == user_data.username:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Username already exists"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already exists"
                    )
            
            # Create permissions
            permissions = []
            for perm_type in user_data.permissions:
                permissions.append(UserPermission(
                    permission_type=perm_type,
                    granted=True,
                    granted_at=datetime.utcnow(),
                    granted_by=created_by
                ))
            
            # Create user document
            user_doc = UserDocument(
                username=user_data.username,
                email=user_data.email,
                full_name=user_data.full_name,
                password_hash=self._hash_password(user_data.password),
                role=user_data.role,
                is_active=user_data.is_active,
                permissions=permissions,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Insert user
            result = await collection.insert_one(user_doc.dict())
            
            # Retrieve and return the created user
            created_user = await collection.find_one({"_id": result.inserted_id})
            return self._document_to_response(created_user)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get a user by ID."""
        try:
            from bson import ObjectId
            collection = await self._get_collection()
            
            user_doc = await collection.find_one({"_id": ObjectId(user_id)})
            if not user_doc:
                return None
            
            return self._document_to_response(user_doc)
            
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[UserResponse]:
        """Get a user by username."""
        try:
            collection = await self._get_collection()
            
            user_doc = await collection.find_one({"username": username})
            if not user_doc:
                return None
            
            return self._document_to_response(user_doc)
            
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[UserResponse]:
        """Get a user by email."""
        try:
            collection = await self._get_collection()
            
            user_doc = await collection.find_one({"email": email})
            if not user_doc:
                return None
            
            return self._document_to_response(user_doc)
            
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            return None
    
    async def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[UserResponse]:
        """Update a user."""
        try:
            from bson import ObjectId
            collection = await self._get_collection()
            
            # Build update data
            update_data = {"updated_at": datetime.utcnow()}
            
            if user_data.username is not None:
                # Check if username already exists
                existing_user = await collection.find_one({
                    "username": user_data.username,
                    "_id": {"$ne": ObjectId(user_id)}
                })
                if existing_user:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Username already exists"
                    )
                update_data["username"] = user_data.username
            
            if user_data.email is not None:
                # Check if email already exists
                existing_user = await collection.find_one({
                    "email": user_data.email,
                    "_id": {"$ne": ObjectId(user_id)}
                })
                if existing_user:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already exists"
                    )
                update_data["email"] = user_data.email
            
            if user_data.full_name is not None:
                update_data["full_name"] = user_data.full_name
            
            if user_data.role is not None:
                update_data["role"] = user_data.role
            
            if user_data.is_active is not None:
                update_data["is_active"] = user_data.is_active
            
            if user_data.password is not None:
                update_data["password_hash"] = self._hash_password(user_data.password)
            
            # Update user
            result = await collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            
            if result.matched_count == 0:
                return None
            
            # Return updated user
            updated_user = await collection.find_one({"_id": ObjectId(user_id)})
            return self._document_to_response(updated_user)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user"
            )
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        try:
            from bson import ObjectId
            collection = await self._get_collection()
            
            result = await collection.delete_one({"_id": ObjectId(user_id)})
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False
    
    async def get_users(
        self,
        skip: int = 0,
        limit: int = 50,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None
    ) -> UserListResponse:
        """Get a list of users with pagination and filtering."""
        try:
            collection = await self._get_collection()
            
            # Build filter
            filter_query = {}
            if role is not None:
                filter_query["role"] = role
            if is_active is not None:
                filter_query["is_active"] = is_active
            
            # Get total count
            total_count = await collection.count_documents(filter_query)
            
            # Get users
            cursor = collection.find(filter_query).sort("created_at", DESCENDING).skip(skip).limit(limit)
            users_docs = await cursor.to_list(length=limit)
            
            # Convert to response objects
            users = [self._document_to_response(doc) for doc in users_docs]
            
            return UserListResponse(
                users=users,
                total_count=total_count,
                page=(skip // limit) + 1,
                page_size=limit,
                has_next=(skip + limit) < total_count
            )
            
        except Exception as e:
            logger.error(f"Error getting users: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve users"
            )
    
    async def update_user_permissions(
        self,
        user_id: str,
        permission_data: UserPermissionUpdate
    ) -> Optional[UserResponse]:
        """Update user permissions."""
        try:
            from bson import ObjectId
            collection = await self._get_collection()
            
            # Create new permissions
            permissions = []
            for perm_type in permission_data.permissions:
                permissions.append(UserPermission(
                    permission_type=perm_type,
                    granted=True,
                    granted_at=datetime.utcnow(),
                    granted_by=permission_data.granted_by
                ).dict())
            
            # Update user permissions
            result = await collection.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$set": {
                        "permissions": permissions,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.matched_count == 0:
                return None
            
            # Return updated user
            updated_user = await collection.find_one({"_id": ObjectId(user_id)})
            return self._document_to_response(updated_user)
            
        except Exception as e:
            logger.error(f"Error updating user permissions: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user permissions"
            )
    
    async def authenticate_user(self, login_data: LoginRequest) -> LoginResponse:
        """Authenticate a user and return a JWT token."""
        try:
            collection = await self._get_collection()
            
            # Find user by username or email
            user_doc = await collection.find_one({
                "$or": [
                    {"username": login_data.username},
                    {"email": login_data.username}
                ]
            })
            
            if not user_doc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password"
                )
            
            # Verify password
            if not self._verify_password(login_data.password, user_doc["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password"
                )
            
            # Check if user is active
            if not user_doc.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account is disabled"
                )
            
            # Update last login
            from bson import ObjectId
            await collection.update_one(
                {"_id": user_doc["_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            
            # Create access token
            token_data = {
                "sub": str(user_doc["_id"]),
                "username": user_doc["username"],
                "role": user_doc["role"]
            }
            access_token = self._create_access_token(token_data)
            
            # Return login response
            user_response = self._document_to_response(user_doc)
            return LoginResponse(
                access_token=access_token,
                token_type="bearer",
                user=user_response,
                expires_in=self.access_token_expire_minutes * 60
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )
    
    async def change_password(
        self,
        user_id: str,
        password_data: PasswordChangeRequest
    ) -> bool:
        """Change user password."""
        try:
            from bson import ObjectId
            collection = await self._get_collection()
            
            # Get current user
            user_doc = await collection.find_one({"_id": ObjectId(user_id)})
            if not user_doc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Verify current password
            if not self._verify_password(password_data.current_password, user_doc["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect"
                )
            
            # Update password
            result = await collection.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$set": {
                        "password_hash": self._hash_password(password_data.new_password),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            return result.modified_count > 0
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error changing password: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to change password"
            )
    
    def _document_to_response(self, doc: Dict[str, Any]) -> UserResponse:
        """Convert a database document to a UserResponse object."""
        return UserResponse(
            id=str(doc["_id"]),
            username=doc["username"],
            email=doc["email"],
            full_name=doc.get("full_name"),
            role=doc["role"],
            is_active=doc["is_active"],
            permissions=[UserPermission(**perm) for perm in doc.get("permissions", [])],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            last_login=doc.get("last_login")
        )


# Dependency function
async def get_user_service() -> UserService:
    """Get user service instance."""
    return UserService()