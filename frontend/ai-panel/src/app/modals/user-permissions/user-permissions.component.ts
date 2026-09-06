import { Component, Inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { MatDialogRef, MAT_DIALOG_DATA, MatDialogModule } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { UserService, User, Permission } from '../../services/user.service';

export interface UserPermissionsDialogData {
  user: User;
}

@Component({
  selector: 'app-user-permissions',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatCheckboxModule,
    MatIconModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './user-permissions.component.html',
  styleUrls: ['./user-permissions.component.scss']
})
export class UserPermissionsComponent implements OnInit {
  form!: FormGroup;
  loading = false;
  user: User;
  
  // Permission types from the backend
  permissionTypes = [
    { value: 'RAG', label: 'RAG' },
    { value: 'LLM', label: 'LLM' },
    { value: 'Conversations', label: 'مکالمات' },
    { value: 'Contribute', label: 'مشارکت' },
    { value: 'Chat', label: 'چت' },
    { value: 'Guide', label: 'راهنما' },
    { value: 'Settings', label: 'تنظیمات' },
    { value: 'Docs', label: 'اسناد' }
  ];

  constructor(
    private fb: FormBuilder,
    public dialogRef: MatDialogRef<UserPermissionsComponent>,
    @Inject(MAT_DIALOG_DATA) public data: UserPermissionsDialogData,
    private userService: UserService
  ) {
    this.user = data.user;
  }

  ngOnInit(): void {
    this.initForm();
  }

  private initForm(): void {
    const formControls: any = {};
    
    // Create a form control for each permission type
    this.permissionTypes.forEach(type => {
      // Check if the user already has this permission granted
      const existingPermission = this.user.permissions?.find(p => p.permission_type === type.value);
      formControls[type.value] = [existingPermission?.granted || false];
    });

    this.form = this.fb.group(formControls);
  }

  onSave(): void {
    if (this.form.valid) {
      this.loading = true;
      
      // Get the selected permissions
      const selectedPermissions: string[] = [];
      Object.keys(this.form.value).forEach(key => {
        if (this.form.value[key]) {
          selectedPermissions.push(key);
        }
      });
      
      this.userService.updateUserPermissions(this.user.id, selectedPermissions)
        .subscribe({
          next: (updatedUser) => {
            this.loading = false;
            this.dialogRef.close(updatedUser);
          },
          error: (error) => {
            console.error('Error updating permissions:', error);
            this.loading = false;
          }
        });
    }
  }

  onCancel(): void {
    this.dialogRef.close();
  }
}