import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { BehaviorSubject, Observable, throwError } from 'rxjs';
import { catchError, tap, map } from 'rxjs/operators';
import { environment } from '../../environments/environment';

// Matches backend schema: UserPermission from user.py
export interface Permission {
  permission_type: string;
  granted: boolean;
  granted_at: string;
  granted_by: string;
}

// Matches backend schema: UserRole from user.py
export type UserRole = 'admin' | 'user' | 'moderator';

// Matches backend schema: UserResponse from user.py
export interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_login?: string;
  permissions: Permission[];
}

// Matches backend schema: UserCreate from user.py
export interface CreateUserRequest {
  username: string;
  email: string;
  full_name: string;
  password?: string; // Handled by backend, might be needed for creation form
  role: 'admin' | 'user' | 'moderator';
}

// Matches backend schema: UserUpdate from user.py
export interface UpdateUserRequest {
  username?: string;
  email?: string;
  full_name?: string;
  role?: 'admin' | 'user' | 'moderator';
  is_active?: boolean;
  password?: string; // For password changes
}

// Matches backend response for GET /users
export interface UserListResponse {
  data: User[];
  total: number;
}

// Filters for GET /users endpoint
export interface UserFilters {
  search?: string;
  role?: UserRole;
  is_active?: boolean;
}

// For PUT /users/{user_id}/permissions
export interface UserPermissionUpdateRequest {
  permissions: string[];
  granted_by?: string;
}

@Injectable({
  providedIn: 'root'
})
export class UserService {
  private apiUrl = `${environment.apiUrl}/api/users`;
  private usersSubject = new BehaviorSubject<User[]>([]);
  public users$ = this.usersSubject.asObservable();

  constructor(private http: HttpClient) {}

  /**
   * Get all users with filtering and pagination.
   * @param filters - Filtering options for role and active status.
   * @param page - The page number to retrieve.
   * @param limit - The number of users per page.
   */
  getUsers(filters: UserFilters = {}, page: number = 1, limit: number = 10): Observable<UserListResponse> {
    const skip = (page - 1) * limit;
    let params = new HttpParams()
      .set('skip', skip.toString())
      .set('limit', limit.toString());

    if (filters.search) {
      params = params.set('search', filters.search);
    }
    if (filters.role) {
      params = params.set('role', filters.role);
    }
    if (filters.is_active !== undefined) {
      params = params.set('is_active', String(filters.is_active));
    }

    return this.http.get<any>(`${this.apiUrl}/`, { params }).pipe(
      map(response => {
        // Map the server response to the expected format
        const mappedResponse: UserListResponse = {
          data: response.users || [],
          total: response.total_count || 0
        };
        this.usersSubject.next(mappedResponse.data);
        return mappedResponse;
      }),
      catchError(this.handleError)
    );
  }

  /**
   * Get a single user by their ID.
   * @param id - The ID of the user.
   */
  getUserById(id: string): Observable<User> {
    return this.http.get<User>(`${this.apiUrl}/${id}`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get the profile of the currently authenticated user.
   */
  getCurrentUserProfile(): Observable<User> {
    return this.http.get<User>(`${this.apiUrl}/me`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Create a new user.
   * @param userData - The data for the new user.
   */
  createUser(userData: CreateUserRequest): Observable<User> {
    return this.http.post<User>(`${this.apiUrl}/`, userData).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Update an existing user.
   * @param id - The ID of the user to update.
   * @param userData - The updated user data.
   */
  updateUser(id: string, userData: UpdateUserRequest): Observable<User> {
    return this.http.put<User>(`${this.apiUrl}/${id}`, userData).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Delete a user by their ID.
   * @param id - The ID of the user to delete.
   */
  deleteUser(id: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/${id}`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Toggle a user's active status.
   * @param id - The ID of the user.
   * @param is_active - The current active status.
   */
  toggleUserStatus(id: string, is_active: boolean): Observable<User> {
    return this.updateUser(id, { is_active: !is_active });
  }

  /**
   * Update a user's permissions.
   * @param id - The ID of the user.
   * @param permissions - An array of permission strings.
   */
  updateUserPermissions(id: string, permissions: string[]): Observable<User> {
    const payload: UserPermissionUpdateRequest = { 
      permissions,
      granted_by: "SYSTEM"
    };
    return this.http.put<User>(`${this.apiUrl}/${id}/permissions`, payload).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * A generic error handler for API calls.
   */
  private handleError(error: any): Observable<never> {
    console.error('An API error occurred', error);
    // In a real-world app, you might send the error to a remote logging infrastructure
    // and/or transform the error for user consumption.
    return throwError(() => new Error('Something bad happened; please try again later.'));
  }
}