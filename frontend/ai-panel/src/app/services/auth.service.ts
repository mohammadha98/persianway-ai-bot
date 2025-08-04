import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { Router } from '@angular/router';

export interface User {
  id: string;
  username: string;
  email: string;
  role: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private readonly STORAGE_KEY = 'auth_user';
  private readonly TOKEN_KEY = 'auth_token';
  
  // Static users for demonstration
  private readonly STATIC_USERS: User[] = [
    {
      id: '1',
      username: 'admin',
      email: 'admin@persianway.com',
      role: 'admin'
    },
    {
      id: '2',
      username: 'manager',
      email: 'manager@persianway.com',
      role: 'manager'
    },
    {
      id: '3',
      username: 'user',
      email: 'user@persianway.com',
      role: 'user'
    }
  ];

  // Static credentials (username: password)
  private readonly STATIC_CREDENTIALS: { [key: string]: string } = {
    'admin': 'admin123',
    'manager': 'manager123',
    'user': 'user123'
  };

  private currentUserSubject = new BehaviorSubject<User | null>(this.getUserFromStorage());
  private isAuthenticatedSubject = new BehaviorSubject<boolean>(this.hasValidToken());

  public currentUser$ = this.currentUserSubject.asObservable();
  public isAuthenticated$ = this.isAuthenticatedSubject.asObservable();

  constructor(private router: Router) {
    // Check if user is already logged in on service initialization
    this.checkAuthStatus();
  }

  /**
   * Login with static credentials
   */
  login(credentials: LoginCredentials): Observable<{ success: boolean; message: string; user?: User }> {
    return new Observable(observer => {
      // Simulate API delay
      setTimeout(() => {
        const { username, password } = credentials;
        
        // Check if credentials are valid
        if (this.STATIC_CREDENTIALS[username] === password) {
          const user = this.STATIC_USERS.find(u => u.username === username);
          
          if (user) {
            // Generate a simple token (in real app, this would come from server)
            const token = this.generateToken(user);
            
            // Store user and token
            this.setUserToStorage(user);
            this.setTokenToStorage(token);
            
            // Update subjects
            this.currentUserSubject.next(user);
            this.isAuthenticatedSubject.next(true);
            
            observer.next({
              success: true,
              message: 'ورود موفقیت‌آمیز بود',
              user: user
            });
          } else {
            observer.next({
              success: false,
              message: 'خطا در سیستم'
            });
          }
        } else {
          observer.next({
            success: false,
            message: 'نام کاربری یا رمز عبور اشتباه است'
          });
        }
        
        observer.complete();
      }, 1000); // Simulate 1 second delay
    });
  }

  /**
   * Logout user
   */
  logout(): void {
    // Clear storage
    localStorage.removeItem(this.STORAGE_KEY);
    localStorage.removeItem(this.TOKEN_KEY);
    
    // Update subjects
    this.currentUserSubject.next(null);
    this.isAuthenticatedSubject.next(false);
    
    // Redirect to login
    this.router.navigate(['/login']);
  }

  /**
   * Get current user
   */
  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return this.isAuthenticatedSubject.value;
  }

  /**
   * Check if user has specific role
   */
  hasRole(role: string): boolean {
    const user = this.getCurrentUser();
    return user ? user.role === role : false;
  }

  /**
   * Check if user has any of the specified roles
   */
  hasAnyRole(roles: string[]): boolean {
    const user = this.getCurrentUser();
    return user ? roles.includes(user.role) : false;
  }

  /**
   * Get authentication token
   */
  getToken(): string | null {
    return localStorage.getItem(this.TOKEN_KEY);
  }

  /**
   * Check authentication status
   */
  private checkAuthStatus(): void {
    const user = this.getUserFromStorage();
    const token = this.getToken();
    
    if (user && token && this.isTokenValid(token)) {
      this.currentUserSubject.next(user);
      this.isAuthenticatedSubject.next(true);
    } else {
      this.logout();
    }
  }

  /**
   * Generate a simple token
   */
  private generateToken(user: User): string {
    const timestamp = Date.now();
    const payload = {
      userId: user.id,
      username: user.username,
      role: user.role,
      exp: timestamp + (24 * 60 * 60 * 1000) // 24 hours
    };
    
    // In a real app, this would be a proper JWT token
    return btoa(JSON.stringify(payload));
  }

  /**
   * Check if token is valid
   */
  private isTokenValid(token: string): boolean {
    try {
      const payload = JSON.parse(atob(token));
      return payload.exp > Date.now();
    } catch {
      return false;
    }
  }

  /**
   * Check if there's a valid token
   */
  private hasValidToken(): boolean {
    const token = this.getToken();
    return token ? this.isTokenValid(token) : false;
  }

  /**
   * Store user to localStorage
   */
  private setUserToStorage(user: User): void {
    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(user));
  }

  /**
   * Get user from localStorage
   */
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

  /**
   * Store token to localStorage
   */
  private setTokenToStorage(token: string): void {
    localStorage.setItem(this.TOKEN_KEY, token);
  }

  /**
   * Get available demo credentials for testing
   */
  getDemoCredentials(): { username: string; password: string; role: string }[] {
    return [
      { username: 'admin', password: 'admin123', role: 'مدیر سیستم' },
      { username: 'manager', password: 'manager123', role: 'مدیر' },
      { username: 'user', password: 'user123', role: 'کاربر' }
    ];
  }
}