import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTableModule } from '@angular/material/table';
import { MatPaginatorModule, PageEvent } from '@angular/material/paginator';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatChipsModule } from '@angular/material/chips';
import { MatDialogModule, MatDialog } from '@angular/material/dialog';
import { MatSnackBarModule, MatSnackBar } from '@angular/material/snack-bar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Subject, takeUntil, debounceTime, distinctUntilChanged } from 'rxjs';

import { UserService, User, UserFilters, UserListResponse, UserRole } from '../../services/user.service';
import { UserDialogComponent } from '../../modals/user-dialog/user-dialog.component';
import { UserPermissionsComponent } from '../../modals/user-permissions/user-permissions.component';


@Component({
  selector: 'app-users',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatTableModule,
    MatPaginatorModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatChipsModule,
    MatDialogModule,
    MatSnackBarModule,
    MatTooltipModule,
    MatProgressSpinnerModule,
    FormsModule,
    ReactiveFormsModule
  ],
  templateUrl: './users.component.html',
  styleUrls: ['./users.component.scss']
})
export class UsersComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();
  
  users: User[] = [];
  loading = false;
  totalUsers = 0;
  currentPage = 1;
  pageSize = 10;
  
  // Filters
  searchTerm = '';
  selectedRole: UserRole | '' = '';
  selectedStatus = '';
  
  // Search subject for debouncing
  private searchSubject = new Subject<string>();
  
  // Table columns
  displayedColumns: string[] = ['avatar', 'name', 'email', 'role', 'status', 'lastLogin', 'actions'];
  
  // Role options
  roleOptions: { value: UserRole | ''; label: string }[] = [
    { value: '', label: 'همه نقش‌ها' },
    { value: 'admin', label: 'مدیر' },
    { value: 'moderator', label: 'ناظر' },
    { value: 'user', label: 'کاربر' }
  ];
  
  // Status options
  statusOptions = [
    { value: '', label: 'همه وضعیت‌ها' },
    { value: 'true', label: 'فعال' },
    { value: 'false', label: 'غیرفعال' }
  ];
  
  constructor(
    private userService: UserService,
    private dialog: MatDialog,
    private snackBar: MatSnackBar
  ) {
    // Setup search debouncing
    this.searchSubject.pipe(
      debounceTime(300),
      distinctUntilChanged(),
      takeUntil(this.destroy$)
    ).subscribe(searchTerm => {
      this.searchTerm = searchTerm;
      this.loadUsers();
    });
  }

  ngOnInit(): void {
    this.loadUsers();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadUsers(): void {
    this.loading = true;
    
    const filters: UserFilters = {
      search: this.searchTerm || undefined,
      role: this.selectedRole || undefined,
      is_active: this.selectedStatus ? this.selectedStatus === 'true' : undefined,
    };

    this.userService.getUsers(filters, this.currentPage, this.pageSize)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: UserListResponse) => {
          this.users = response.data;
          this.totalUsers = response.total;
          this.loading = false;
        },
        error: (error: any) => {
          console.error('Error loading users:', error);
          this.showSnackBar('خطا در بارگذاری کاربران', 'error');
          this.loading = false;
        }
      });
  }

  onSearchChange(searchTerm: string): void {
    this.searchSubject.next(searchTerm);
  }

  onFilterChange(): void {
    this.currentPage = 1;
    this.loadUsers();
  }

  onPageChange(event: PageEvent): void {
    this.currentPage = event.pageIndex + 1;
    this.pageSize = event.pageSize;
    this.loadUsers();
  }

  clearFilters(): void {
    this.searchTerm = '';
    this.selectedRole = '';
    this.selectedStatus = '';
    this.currentPage = 1;
    this.loadUsers();
  }

  openCreateUserDialog(): void {
    const dialogRef = this.dialog.open(UserDialogComponent, {
      width: '600px',
      data: { mode: 'create' },
      direction: 'rtl'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.userService.createUser(result)
          .pipe(takeUntil(this.destroy$))
          .subscribe({
            next: () => {
              this.showSnackBar('کاربر با موفقیت ایجاد شد', 'success');
              this.loadUsers();
            },
            error: (error: any) => {
              console.error('Error creating user:', error);
              this.showSnackBar('خطا در ایجاد کاربر', 'error');
            }
          });
      }
    });
  }

  openEditUserDialog(user: User): void {
    const dialogRef = this.dialog.open(UserDialogComponent, {
      width: '600px',
      data: { mode: 'edit', user: { ...user } },
      direction: 'rtl'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.userService.updateUser(user.id, result)
          .pipe(takeUntil(this.destroy$))
          .subscribe({
            next: () => {
              this.showSnackBar('کاربر با موفقیت به‌روزرسانی شد', 'success');
              this.loadUsers();
            },
            error: (error: any) => {
              console.error('Error updating user:', error);
              this.showSnackBar('خطا در به‌روزرسانی کاربر', 'error');
            }
          });
      }
    });
  }

  toggleUserStatus(user: User): void {
    this.userService.toggleUserStatus(user.id, user.is_active)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          const status = user.is_active ? 'غیرفعال' : 'فعال';
          this.showSnackBar(`کاربر با موفقیت ${status} شد`, 'success');
          this.loadUsers();
        },
        error: (error: any) => {
          console.error('Error toggling user status:', error);
          this.showSnackBar('خطا در تغییر وضعیت کاربر', 'error');
        }
      });
  }

  deleteUser(user: User): void {
    if (confirm(`آیا از حذف کاربر "${user.full_name}" اطمینان دارید؟`)) {
      this.userService.deleteUser(user.id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: () => {
            this.showSnackBar('کاربر با موفقیت حذف شد', 'success');
            this.loadUsers();
          },
          error: (error: any) => {
            console.error('Error deleting user:', error);
            this.showSnackBar('خطا در حذف کاربر', 'error');
          }
        });
    }
  }

  openPermissionsDialog(user: User): void {
    const dialogRef = this.dialog.open(UserPermissionsComponent, {
      width: '600px',
      data: { user: { ...user } },
      direction: 'rtl'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.showSnackBar('دسترسی‌های کاربر با موفقیت به‌روزرسانی شد', 'success');
        this.loadUsers();
      }
    });
  }

  getRoleLabel(role: string): string {
    const roleMap: { [key: string]: string } = {
      'admin': 'مدیر',
      'moderator': 'ناظر',
      'user': 'کاربر'
    };
    return roleMap[role] || role;
  }

  getRoleColor(role: string): string {
    const colorMap: { [key: string]: string } = {
      'admin': 'primary',
      'moderator': 'accent',
      'user': 'basic'
    };
    return colorMap[role] || 'basic';
  }

  getStatusColor(is_active: boolean): string {
    return is_active ? 'primary' : 'warn';
  }

  getStatusLabel(is_active: boolean): string {
    return is_active ? 'فعال' : 'غیرفعال';
  }

  formatDate(date: Date | string): string {
    if (!date) return '-';
    const d = new Date(date);
    return new Intl.DateTimeFormat('fa-IR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    }).format(d);
  }

  getInitials(fullName: string): string {
    if (!fullName) return '';
    const names = fullName.split(' ');
    const firstName = names[0];
    const lastName = names.length > 1 ? names[names.length - 1] : '';
    return `${firstName?.charAt(0) || ''}${lastName?.charAt(0) || ''}`.toUpperCase();
  }

  private showSnackBar(message: string, type: 'success' | 'error' | 'info' = 'info'): void {
    this.snackBar.open(message, 'بستن', {
      duration: 3000,
      horizontalPosition: 'center',
      verticalPosition: 'top',
      panelClass: [`snackbar-${type}`],
      direction: 'rtl'
    });
  }
}