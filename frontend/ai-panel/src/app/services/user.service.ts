import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { delay, map } from 'rxjs/operators';

export interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'admin' | 'user' | 'moderator';
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
  lastLogin?: Date;
  avatar?: string;
  phone?: string;
  department?: string;
}

export interface CreateUserRequest {
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'admin' | 'user' | 'moderator';
  phone?: string;
  department?: string;
}

export interface UpdateUserRequest {
  username?: string;
  email?: string;
  firstName?: string;
  lastName?: string;
  role?: 'admin' | 'user' | 'moderator';
  isActive?: boolean;
  phone?: string;
  department?: string;
}

export interface UserListResponse {
  users: User[];
  total: number;
  page: number;
  limit: number;
}

export interface UserFilters {
  search?: string;
  role?: string;
  isActive?: boolean;
  department?: string;
}

@Injectable({
  providedIn: 'root'
})
export class UserService {
  private usersSubject = new BehaviorSubject<User[]>([]);
  public users$ = this.usersSubject.asObservable();

  // Static data for demonstration - will be replaced with API calls
  private staticUsers: User[] = [
    {
      id: '1',
      username: 'admin',
      email: 'admin@persianway.ai',
      firstName: 'مدیر',
      lastName: 'سیستم',
      role: 'admin',
      isActive: true,
      createdAt: new Date('2024-01-01'),
      updatedAt: new Date('2024-01-01'),
      lastLogin: new Date('2024-01-15'),
      department: 'مدیریت',
      phone: '+98-912-345-6789'
    },
    {
      id: '2',
      username: 'user1',
      email: 'user1@persianway.ai',
      firstName: 'علی',
      lastName: 'احمدی',
      role: 'user',
      isActive: true,
      createdAt: new Date('2024-01-02'),
      updatedAt: new Date('2024-01-10'),
      lastLogin: new Date('2024-01-14'),
      department: 'فناوری',
      phone: '+98-912-345-6790'
    },
    {
      id: '3',
      username: 'moderator1',
      email: 'moderator@persianway.ai',
      firstName: 'فاطمه',
      lastName: 'محمدی',
      role: 'moderator',
      isActive: true,
      createdAt: new Date('2024-01-03'),
      updatedAt: new Date('2024-01-12'),
      lastLogin: new Date('2024-01-13'),
      department: 'محتوا',
      phone: '+98-912-345-6791'
    },
    {
      id: '4',
      username: 'user2',
      email: 'user2@persianway.ai',
      firstName: 'محمد',
      lastName: 'رضایی',
      role: 'user',
      isActive: false,
      createdAt: new Date('2024-01-04'),
      updatedAt: new Date('2024-01-11'),
      lastLogin: new Date('2024-01-12'),
      department: 'پشتیبانی',
      phone: '+98-912-345-6792'
    },
    {
      id: '5',
      username: 'user3',
      email: 'user3@persianway.ai',
      firstName: 'زهرا',
      lastName: 'کریمی',
      role: 'user',
      isActive: true,
      createdAt: new Date('2024-01-05'),
      updatedAt: new Date('2024-01-13'),
      lastLogin: new Date('2024-01-15'),
      department: 'تحقیق و توسعه',
      phone: '+98-912-345-6793'
    }
  ];

  constructor() {
    // Initialize with static data
    this.usersSubject.next([...this.staticUsers]);
  }

  /**
   * Get all users with optional filtering and pagination
   */
  getUsers(page: number = 1, limit: number = 10, filters?: UserFilters): Observable<UserListResponse> {
    return of(this.staticUsers).pipe(
      map(users => {
        let filteredUsers = [...users];

        // Apply filters
        if (filters) {
          if (filters.search) {
            const searchTerm = filters.search.toLowerCase();
            filteredUsers = filteredUsers.filter(user => 
              user.username.toLowerCase().includes(searchTerm) ||
              user.email.toLowerCase().includes(searchTerm) ||
              user.firstName.toLowerCase().includes(searchTerm) ||
              user.lastName.toLowerCase().includes(searchTerm)
            );
          }

          if (filters.role) {
            filteredUsers = filteredUsers.filter(user => user.role === filters.role);
          }

          if (filters.isActive !== undefined) {
            filteredUsers = filteredUsers.filter(user => user.isActive === filters.isActive);
          }

          if (filters.department) {
            filteredUsers = filteredUsers.filter(user => 
              user.department?.toLowerCase().includes(filters.department!.toLowerCase())
            );
          }
        }

        // Apply pagination
        const startIndex = (page - 1) * limit;
        const endIndex = startIndex + limit;
        const paginatedUsers = filteredUsers.slice(startIndex, endIndex);

        return {
          users: paginatedUsers,
          total: filteredUsers.length,
          page,
          limit
        };
      }),
      delay(500) // Simulate API delay
    );
  }

  /**
   * Get user by ID
   */
  getUserById(id: string): Observable<User | null> {
    return of(this.staticUsers.find(user => user.id === id) || null).pipe(
      delay(300)
    );
  }

  /**
   * Create a new user
   */
  createUser(userData: CreateUserRequest): Observable<User> {
    const newUser: User = {
      id: this.generateId(),
      ...userData,
      isActive: true,
      createdAt: new Date(),
      updatedAt: new Date()
    };

    this.staticUsers.push(newUser);
    this.usersSubject.next([...this.staticUsers]);

    return of(newUser).pipe(delay(500));
  }

  /**
   * Update an existing user
   */
  updateUser(id: string, userData: UpdateUserRequest): Observable<User | null> {
    const userIndex = this.staticUsers.findIndex(user => user.id === id);
    
    if (userIndex === -1) {
      return of(null).pipe(delay(300));
    }

    const updatedUser: User = {
      ...this.staticUsers[userIndex],
      ...userData,
      updatedAt: new Date()
    };

    this.staticUsers[userIndex] = updatedUser;
    this.usersSubject.next([...this.staticUsers]);

    return of(updatedUser).pipe(delay(500));
  }

  /**
   * Delete a user
   */
  deleteUser(id: string): Observable<boolean> {
    const userIndex = this.staticUsers.findIndex(user => user.id === id);
    
    if (userIndex === -1) {
      return of(false).pipe(delay(300));
    }

    this.staticUsers.splice(userIndex, 1);
    this.usersSubject.next([...this.staticUsers]);

    return of(true).pipe(delay(500));
  }

  /**
   * Toggle user active status
   */
  toggleUserStatus(id: string): Observable<User | null> {
    const user = this.staticUsers.find(user => user.id === id);
    
    if (!user) {
      return of(null).pipe(delay(300));
    }

    return this.updateUser(id, { isActive: !user.isActive });
  }

  /**
   * Get users by role
   */
  getUsersByRole(role: 'admin' | 'user' | 'moderator'): Observable<User[]> {
    return of(this.staticUsers.filter(user => user.role === role)).pipe(
      delay(300)
    );
  }

  /**
   * Get active users count
   */
  getActiveUsersCount(): Observable<number> {
    return of(this.staticUsers.filter(user => user.isActive).length).pipe(
      delay(200)
    );
  }

  /**
   * Get users statistics
   */
  getUsersStatistics(): Observable<{
    total: number;
    active: number;
    inactive: number;
    admins: number;
    users: number;
    moderators: number;
  }> {
    const stats = {
      total: this.staticUsers.length,
      active: this.staticUsers.filter(user => user.isActive).length,
      inactive: this.staticUsers.filter(user => !user.isActive).length,
      admins: this.staticUsers.filter(user => user.role === 'admin').length,
      users: this.staticUsers.filter(user => user.role === 'user').length,
      moderators: this.staticUsers.filter(user => user.role === 'moderator').length
    };

    return of(stats).pipe(delay(300));
  }

  /**
   * Search users by term
   */
  searchUsers(searchTerm: string): Observable<User[]> {
    if (!searchTerm.trim()) {
      return of([]);
    }

    const term = searchTerm.toLowerCase();
    const results = this.staticUsers.filter(user => 
      user.username.toLowerCase().includes(term) ||
      user.email.toLowerCase().includes(term) ||
      user.firstName.toLowerCase().includes(term) ||
      user.lastName.toLowerCase().includes(term) ||
      user.department?.toLowerCase().includes(term)
    );

    return of(results).pipe(delay(300));
  }

  /**
   * Validate if username is available
   */
  isUsernameAvailable(username: string, excludeId?: string): Observable<boolean> {
    const exists = this.staticUsers.some(user => 
      user.username.toLowerCase() === username.toLowerCase() && 
      user.id !== excludeId
    );
    
    return of(!exists).pipe(delay(200));
  }

  /**
   * Validate if email is available
   */
  isEmailAvailable(email: string, excludeId?: string): Observable<boolean> {
    const exists = this.staticUsers.some(user => 
      user.email.toLowerCase() === email.toLowerCase() && 
      user.id !== excludeId
    );
    
    return of(!exists).pipe(delay(200));
  }

  /**
   * Generate a unique ID for new users
   */
  private generateId(): string {
    return Math.max(...this.staticUsers.map(user => parseInt(user.id)), 0) + 1 + '';
  }

  /**
   * Reset to initial data (for testing purposes)
   */
  resetData(): void {
    this.staticUsers = [
      {
        id: '1',
        username: 'admin',
        email: 'admin@persianway.ai',
        firstName: 'مدیر',
        lastName: 'سیستم',
        role: 'admin',
        isActive: true,
        createdAt: new Date('2024-01-01'),
        updatedAt: new Date('2024-01-01'),
        lastLogin: new Date('2024-01-15'),
        department: 'مدیریت',
        phone: '+98-912-345-6789'
      },
      {
        id: '2',
        username: 'user1',
        email: 'user1@persianway.ai',
        firstName: 'علی',
        lastName: 'احمدی',
        role: 'user',
        isActive: true,
        createdAt: new Date('2024-01-02'),
        updatedAt: new Date('2024-01-10'),
        lastLogin: new Date('2024-01-14'),
        department: 'فناوری',
        phone: '+98-912-345-6790'
      },
      {
        id: '3',
        username: 'moderator1',
        email: 'moderator@persianway.ai',
        firstName: 'فاطمه',
        lastName: 'محمدی',
        role: 'moderator',
        isActive: true,
        createdAt: new Date('2024-01-03'),
        updatedAt: new Date('2024-01-12'),
        lastLogin: new Date('2024-01-13'),
        department: 'محتوا',
        phone: '+98-912-345-6791'
      },
      {
        id: '4',
        username: 'user2',
        email: 'user2@persianway.ai',
        firstName: 'محمد',
        lastName: 'رضایی',
        role: 'user',
        isActive: false,
        createdAt: new Date('2024-01-04'),
        updatedAt: new Date('2024-01-11'),
        lastLogin: new Date('2024-01-12'),
        department: 'پشتیبانی',
        phone: '+98-912-345-6792'
      },
      {
        id: '5',
        username: 'user3',
        email: 'user3@persianway.ai',
        firstName: 'زهرا',
        lastName: 'کریمی',
        role: 'user',
        isActive: true,
        createdAt: new Date('2024-01-05'),
        updatedAt: new Date('2024-01-13'),
        lastLogin: new Date('2024-01-15'),
        department: 'تحقیق و توسعه',
        phone: '+98-912-345-6793'
      }
    ];
    this.usersSubject.next([...this.staticUsers]);
  }
}