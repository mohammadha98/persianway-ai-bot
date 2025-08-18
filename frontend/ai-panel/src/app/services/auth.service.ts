import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { catchError, map } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export enum PermissionType {
  RAG = 'RAG',
  LLM = 'LLM',
  Conversations = 'Conversations',
  Contribute = 'Contribute',
  Chat = 'Chat',
  Guide = 'Guide',
  Settings = 'Settings',
  Docs = 'Docs'
}

export interface Permission {
  permission_type: PermissionType | string;
  granted: boolean;
  granted_at: string;
  granted_by: string;
}

export interface User {
  id: string;
  username: string;
  email: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
  permissions: Permission[];
  created_at: string;
  updated_at: string;
  last_login: string | null;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
  expires_in: number;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private readonly STORAGE_KEY = 'auth_user';
  private readonly TOKEN_KEY = 'auth_token';
  private apiUrl = environment.apiUrl;

  private currentUserSubject = new BehaviorSubject<User | null>(this.getUserFromStorage());
  private isAuthenticatedSubject = new BehaviorSubject<boolean>(this.hasValidToken());

  public currentUser$ = this.currentUserSubject.asObservable();
  public isAuthenticated$ = this.isAuthenticatedSubject.asObservable();

  constructor(private router: Router, private http: HttpClient) {
    this.checkAuthStatus();
  }

  login(credentials: LoginCredentials): Observable<{ success: boolean; message: string; user?: User }> {
    return this.http.post<LoginResponse>(`${this.apiUrl}/api/users/login`, credentials).pipe(
      map(response => {
        this.setUserToStorage(response.user);
        this.setTokenToStorage(response.access_token);
        this.currentUserSubject.next(response.user);
        this.isAuthenticatedSubject.next(true);
        return {
          success: true,
          message: 'ورود موفقیت‌آمیز بود',
          user: response.user
        };
      }),
      catchError(error => {
        let errorMessage = 'نام کاربری یا رمز عبور اشتباه است';
        if (error.error && error.error.detail) {
          if (typeof error.error.detail === 'string') {
            errorMessage = error.error.detail;
          } else if (Array.isArray(error.error.detail)) {
            errorMessage = error.error.detail.map((err:any) => err.msg).join(', ');
          }
        }
        return of({
          success: false,
          message: errorMessage
        });
      })
    );
  }

  logout(): void {
    localStorage.removeItem(this.STORAGE_KEY);
    localStorage.removeItem(this.TOKEN_KEY);
    this.currentUserSubject.next(null);
    this.isAuthenticatedSubject.next(false);
    this.router.navigate(['/login']);
  }

  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }

  isAuthenticated(): boolean {
    return this.isAuthenticatedSubject.value;
  }

  hasRole(role: string): boolean {
    const user = this.getCurrentUser();
    return user ? user.role === role : false;
  }

  hasAnyRole(roles: string[]): boolean {
    const user = this.getCurrentUser();
    return user ? roles.includes(user.role) : false;
  }

  hasPermission(permissionType: PermissionType): boolean {
    const user = this.getCurrentUser();
    if (!user || !user.permissions) return false;
    
    return user.permissions.some(permission => 
      permission.permission_type === permissionType && permission.granted
    );
  }

  hasAnyPermission(permissionTypes: PermissionType[]): boolean {
    const user = this.getCurrentUser();
    if (!user || !user.permissions) return false;
    
    return permissionTypes.some(type => this.hasPermission(type));
  }

  hasAllPermissions(permissionTypes: PermissionType[]): boolean {
    const user = this.getCurrentUser();
    if (!user || !user.permissions) return false;
    
    return permissionTypes.every(type => this.hasPermission(type));
  }

  getToken(): string | null {
    return localStorage.getItem(this.TOKEN_KEY);
  }

  private checkAuthStatus(): void {
    const user = this.getUserFromStorage();
    const token = this.getToken();
    if (user && token) {
      this.currentUserSubject.next(user);
      this.isAuthenticatedSubject.next(true);
    } else {
      this.currentUserSubject.next(null);
      this.isAuthenticatedSubject.next(false);
    }
  }

  private hasValidToken(): boolean {
    return !!this.getToken();
  }

  private setUserToStorage(user: User): void {
    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(user));
  }

  private getUserFromStorage(): User | null {
    const userStr = localStorage.getItem(this.STORAGE_KEY);
    if (userStr) {
      try {
        return JSON.parse(userStr);
      } catch {
        return null;
      }
    }
    return null;
  }

  private setTokenToStorage(token: string): void {
    localStorage.setItem(this.TOKEN_KEY, token);
  }

  /**
   * Returns all available permission types as an array
   */
  getAllPermissionTypes(): PermissionType[] {
    return Object.values(PermissionType);
  }

  /**
   * Returns all granted permissions for the current user
   */
  getUserGrantedPermissions(): PermissionType[] {
    const user = this.getCurrentUser();
    if (!user || !user.permissions) return [];
    
    return user.permissions
      .filter(permission => permission.granted)
      .map(permission => permission.permission_type as PermissionType);
  }
}