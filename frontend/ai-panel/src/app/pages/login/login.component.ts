import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { CommonModule } from '@angular/common';
import { AuthService, LoginCredentials } from '../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatCheckboxModule,
    MatSnackBarModule,
    MatProgressSpinnerModule,
    ReactiveFormsModule
  ],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent implements OnInit {
  loginForm!: FormGroup;
  hidePassword = true;
  isLoading = false;
  demoCredentials: { username: string; password: string; role: string }[] = [];

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.initializeForm();
    this.loadDemoCredentials();
    
    // Redirect if already authenticated
    if (this.authService.isAuthenticated()) {
      this.router.navigate(['/dashboard']);
    }
  }

  private initializeForm(): void {
    this.loginForm = this.fb.group({
      username: ['', [Validators.required, Validators.minLength(3)]],
      password: ['', [Validators.required, Validators.minLength(6)]],
      rememberMe: [false]
    });
  }

  private loadDemoCredentials(): void {
    this.demoCredentials = this.authService.getDemoCredentials();
  }

  onSubmit(): void {
    if (this.loginForm.valid && !this.isLoading) {
      this.isLoading = true;
      
      const credentials: LoginCredentials = {
        username: this.loginForm.value.username,
        password: this.loginForm.value.password
      };

      this.authService.login(credentials).subscribe({
        next: (response) => {
          this.isLoading = false;
          
          if (response.success) {
            this.snackBar.open(response.message, 'بستن', {
              duration: 3000,
              direction: 'rtl',
              panelClass: ['success-snackbar']
            });
            
            // Redirect to home or intended route
            this.router.navigate(['/']);
          } else {
            this.snackBar.open(response.message, 'بستن', {
              duration: 5000,
              direction: 'rtl',
              panelClass: ['error-snackbar']
            });
          }
        },
        error: (error) => {
          this.isLoading = false;
          this.snackBar.open('خطا در اتصال به سرور', 'بستن', {
            duration: 5000,
            direction: 'rtl',
            panelClass: ['error-snackbar']
          });
        }
      });
    } else {
      this.markFormGroupTouched();
    }
  }

  private markFormGroupTouched(): void {
    Object.keys(this.loginForm.controls).forEach(key => {
      const control = this.loginForm.get(key);
      control?.markAsTouched();
    });
  }

  getErrorMessage(fieldName: string): string {
    const control = this.loginForm.get(fieldName);
    
    if (control?.hasError('required')) {
      return `${this.getFieldDisplayName(fieldName)} الزامی است`;
    }
    
    if (control?.hasError('minlength')) {
      const requiredLength = control.errors?.['minlength']?.requiredLength;
      return `${this.getFieldDisplayName(fieldName)} باید حداقل ${requiredLength} کاراکتر باشد`;
    }
    
    return '';
  }

  private getFieldDisplayName(fieldName: string): string {
    const fieldNames: { [key: string]: string } = {
      username: 'نام کاربری',
      password: 'رمز عبور'
    };
    
    return fieldNames[fieldName] || fieldName;
  }

  fillDemoCredentials(credentials: { username: string; password: string }): void {
    this.loginForm.patchValue({
      username: credentials.username,
      password: credentials.password
    });
  }

  onForgotPassword(): void {
    this.snackBar.open('این قابلیت در حال توسعه است', 'بستن', {
      duration: 3000,
      direction: 'rtl'
    });
  }
}
