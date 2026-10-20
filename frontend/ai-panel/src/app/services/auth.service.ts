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
  Docs = 'Docs',
  Analysis = 'Analysis'
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
  /** Expiry of TOKEN_KEY in epoch milliseconds, read from the JWT `exp`. */
  private readonly TOKEN_EXPIRY_KEY = 'auth_token_expires_at';
  private apiUrl = environment.apiUrl;

  /** Fires exactly when the stored token expires; see scheduleExpiryLogout(). */
  private expiryTimer: ReturnType<typeof setTimeout> | null = null;

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
        this.setTokenToStorage(response.access_token, response.expires_in);
        this.currentUserSubject.next(response.user);
        this.isAuthenticatedSubject.next(true);
        this.scheduleExpiryLogout();
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
    this.clearSession();
    this.router.navigate(['/login']);
  }

  /**
   * End the session because the token is no longer usable (expired, rejected
   * with 401, or the account was disabled) and send the operator to the login
   * page. Safe to call repeatedly: once the session is gone it only clears.
   *
   * Without this the panel kept the previous user on screen, every API call
   * answered 401, and the only way out was a manual sign-out.
   */
  expireSession(): void {
    const hadSession = !!this.getToken() || !!this.currentUserSubject.value;
    this.clearSession();
    if (hadSession) {
      this.router.navigate(['/login'], { queryParams: { expired: '1' } });
    }
  }

  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }

  isAuthenticated(): boolean {
    // Re-check the clock: a component that is still mounted must not look
    // "signed in" after `exp` has passed, even if the timer was throttled.
    return this.isAuthenticatedSubject.value && this.hasValidToken();
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
    if (user && token && !this.isTokenExpired()) {
      this.currentUserSubject.next(user);
      this.isAuthenticatedSubject.next(true);
      this.scheduleExpiryLogout();
    } else {
      // Nothing stored, or the token has already expired: drop the stale user
      // so a reload cannot put the panel back into a dead session.
      this.clearSession();
    }
  }

  /** True only when a token is stored and its `exp` claim is still ahead. */
  private hasValidToken(): boolean {
    return !!this.getToken() && !this.isTokenExpired();
  }

  private isTokenExpired(): boolean {
    const expiry = this.getTokenExpiry();
    if (expiry === null) {
      // No readable `exp` (a non-JWT token, or one issued before the expiry
      // key existed). Presence is the only signal available; a 401 from the
      // API still ends the session through the interceptor.
      return false;
    }
    return Date.now() >= expiry;
  }

  private getTokenExpiry(): number | null {
    const stored = Number(localStorage.getItem(this.TOKEN_EXPIRY_KEY));
    if (stored > 0) {
      return stored;
    }
    return this.decodeTokenExpiry(this.getToken());
  }

  /** Convert a JWT's `exp` claim (seconds) to epoch milliseconds. */
  private decodeTokenExpiry(token: string | null): number | null {
    if (!token) {
      return null;
    }
    const parts = token.split('.');
    if (parts.length !== 3) {
      return null;
    }
    try {
      const payload = JSON.parse(this.decodeBase64Url(parts[1]));
      return typeof payload?.exp === 'number' ? payload.exp * 1000 : null;
    } catch {
      return null;
    }
  }

  private decodeBase64Url(segment: string): string {
    const base64 = segment.replace(/-/g, '+').replace(/_/g, '/');
    const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, '=');
    // `atob` returns a byte string; the payload is UTF-8 JSON, so rebuild the
    // bytes before decoding or a non-ASCII claim would throw.
    const bytes = atob(padded);
    return decodeURIComponent(
      bytes.split('').map(char => '%' + char.charCodeAt(0).toString(16).padStart(2, '0')).join('')
    );
  }

  /**
   * Arm a timer for the exact moment the token expires so an idle operator is
   * signed out on time instead of being left on a page whose requests all
   * answer 401.
   */
  private scheduleExpiryLogout(): void {
    this.cancelExpiryTimer();
    const expiry = this.getTokenExpiry();
    if (expiry === null) {
      return;
    }
    const remaining = expiry - Date.now();
    if (remaining <= 0) {
      this.expireSession();
      return;
    }
    // setTimeout keeps its delay in a signed 32-bit integer; anything larger
    // fires immediately. Three days is well below it, but clamp anyway so a
    // longer configured lifetime cannot break the timer.
    const maxDelay = 2 ** 31 - 1;
    this.expiryTimer = setTimeout(() => {
      this.expiryTimer = null;
      if (this.isTokenExpired()) {
        this.expireSession();
      } else {
        // Woke up early (clock adjustment / timer clamping): re-arm.
        this.scheduleExpiryLogout();
      }
    }, Math.min(remaining, maxDelay));
  }

  private cancelExpiryTimer(): void {
    if (this.expiryTimer !== null) {
      clearTimeout(this.expiryTimer);
      this.expiryTimer = null;
    }
  }

  /** Forget the user and token; does not navigate. */
  private clearSession(): void {
    this.cancelExpiryTimer();
    localStorage.removeItem(this.STORAGE_KEY);
    localStorage.removeItem(this.TOKEN_KEY);
    localStorage.removeItem(this.TOKEN_EXPIRY_KEY);
    this.currentUserSubject.next(null);
    this.isAuthenticatedSubject.next(false);
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

  private setTokenToStorage(token: string, expiresInSeconds?: number): void {
    localStorage.setItem(this.TOKEN_KEY, token);
    // The JWT `exp` is authoritative; `expires_in` is the fallback for a token
    // whose payload is not a decodable JWT.
    let expiry = this.decodeTokenExpiry(token);
    if (expiry === null && expiresInSeconds && expiresInSeconds > 0) {
      expiry = Date.now() + expiresInSeconds * 1000;
    }
    if (expiry !== null) {
      localStorage.setItem(this.TOKEN_EXPIRY_KEY, String(expiry));
    } else {
      localStorage.removeItem(this.TOKEN_EXPIRY_KEY);
    }
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