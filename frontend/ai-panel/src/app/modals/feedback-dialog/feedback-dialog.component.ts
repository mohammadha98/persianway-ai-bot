import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatDialogModule, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { MatTabsModule } from '@angular/material/tabs';
import { MatRadioModule } from '@angular/material/radio';

export interface FeedbackDialogData {
  messageContent: string;
  question: string;
  messageId: string;
  conversationId?: string;
  mode?: 'report' | 'edit';  // Type of feedback being given
}

export interface FeedbackDialogResult {
  feedback: 'report' | 'edit' | 'approve';
  comment: string;
  category?: string;
  edited_answer?: string;
}

@Component({
  selector: 'app-feedback-dialog',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatButtonModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    MatDialogModule,
    MatTabsModule,
    MatRadioModule
  ],
  templateUrl: './feedback-dialog.component.html',
  styleUrl: './feedback-dialog.component.scss'
})
export class FeedbackDialogComponent {
  comment = '';
  selectedCategory = 'incorrect_info';
  editedAnswer = '';
  mode: 'report' | 'edit' = 'report';

  categories = [
    { value: 'incorrect_info', label: 'اطلاعات نادرست' },
    { value: 'incomplete', label: 'پاسخ ناقص' },
    { value: 'irrelevant', label: 'پاسخ نامرتبط' },
    { value: 'outdated', label: 'اطلاعات قدیمی' },
    { value: 'harmful', label: 'محتوای نامناسب' },
    { value: 'grammar', label: 'مشکل نگارشی' },
    { value: 'other', label: 'سایر موارد' }
  ];

  constructor(
    public dialogRef: MatDialogRef<FeedbackDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: FeedbackDialogData
  ) {
    // Set mode from data, default to 'report'
    this.mode = data.mode || 'report';
    // Pre-fill edited answer with the original content for edit mode
    if (this.mode === 'edit') {
      this.editedAnswer = data.messageContent;
    }
  }

  onCancel(): void {
    this.dialogRef.close();
  }

  onSubmit(): void {
    // Validate based on mode
    if (this.mode === 'report') {
      if (!this.comment.trim()) {
        return;
      }
      this.dialogRef.close({
        feedback: 'report',
        comment: this.comment,
        category: this.selectedCategory
      } as FeedbackDialogResult);
    } else if (this.mode === 'edit') {
      if (!this.editedAnswer.trim() || !this.comment.trim()) {
        return;
      }
      this.dialogRef.close({
        feedback: 'edit',
        comment: this.comment,
        edited_answer: this.editedAnswer
      } as FeedbackDialogResult);
    }
  }

  get isReportMode(): boolean {
    return this.mode === 'report';
  }

  get isEditMode(): boolean {
    return this.mode === 'edit';
  }
}