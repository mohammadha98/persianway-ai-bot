import { Component, Inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { MatDialogRef, MAT_DIALOG_DATA, MatDialogModule } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatSelectModule } from '@angular/material/select';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatIconModule } from '@angular/material/icon';
import { UserService } from '../../services/user.service';

export interface UserDialogData {
  mode: 'create' | 'edit';
  user?: any;
}

@Component({
  selector: 'app-user-dialog',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatSelectModule,
    MatSlideToggleModule,
    MatIconModule
  ],
  templateUrl: './user-dialog.component.html',
  styleUrls: ['./user-dialog.component.scss']
})
export class UserDialogComponent implements OnInit {
  form!: FormGroup;
  isEditMode = false;

  roleOptions = [
    { value: 'admin', label: 'مدیر' },
    { value: 'user', label: 'کاربر' }
  ];

  constructor(
    private fb: FormBuilder,
    public dialogRef: MatDialogRef<UserDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: UserDialogData,
    private userService: UserService
  ) { }

  ngOnInit(): void {
    this.isEditMode = this.data.mode === 'edit';
    this.initForm();
  }

  private initForm(): void {
    this.form = this.fb.group({
      firstName: [this.data.user?.firstName || '', Validators.required],
      lastName: [this.data.user?.lastName || '', Validators.required],
      username: [this.data.user?.username || '', [Validators.required, Validators.pattern(/^[a-zA-Z0-9_]+$/)]],
      email: [this.data.user?.email || '', [Validators.required, Validators.email]],
      department: [this.data.user?.department || ''],
      role: [this.data.user?.role || 'user', Validators.required],
      isActive: [this.isEditMode ? this.data.user?.isActive : true, Validators.required],
      password: ['', this.isEditMode ? [] : [Validators.required, Validators.minLength(8)]]
    });
  }

  onSave(): void {
    if (this.form.valid) {
      this.dialogRef.close(this.form.value);
    }
  }

  onCancel(): void {
    this.dialogRef.close();
  }
}