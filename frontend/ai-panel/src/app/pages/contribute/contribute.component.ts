import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatSnackBarModule, MatSnackBar } from '@angular/material/snack-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ContributeService } from '../../services/contribute.service';

interface ContributionForm {
  title: string;
  content: string;
  category: string;
  tags: string;
  source: string;
  author: string;
  additionalReferences: string;
}

@Component({
  selector: 'app-contribute',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatInputModule,
    MatFormFieldModule,
    MatSelectModule,
    MatSnackBarModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './contribute.component.html',
  styleUrl: './contribute.component.scss'
})
export class ContributeComponent {
  form: ContributionForm = {
    title: '',
    content: '',
    category: '',
    tags: '',
    source: '',
    author: '',
    additionalReferences: ''
  };

  categories = [
    { value: 'crops', label: 'زراعت و محصولات زراعی' },
    { value: 'livestock', label: 'دامداری و طیور' },
    { value: 'horticulture', label: 'باغبانی و گلخانه' },
    { value: 'soil', label: 'خاک و کود' },
    { value: 'pest', label: 'آفات و بیماری‌ها' },
    { value: 'irrigation', label: 'آبیاری و منابع آب' },
    { value: 'machinery', label: 'ماشین‌آلات کشاورزی' },
    { value: 'marketing', label: 'بازاریابی و فروش' },
    { value: 'other', label: 'سایر موضوعات' }
  ];

  isSubmitting = false;
  selectedFile: File | null = null;

  constructor(
    private contributeService: ContributeService,
    private snackBar: MatSnackBar
  ) { }

  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      // Check file type
      const allowedTypes = ['application/pdf', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
      if (allowedTypes.includes(file.type)) {
        this.selectedFile = file;
        this.showMessage('فایل انتخاب شد: ' + file.name);
      } else {
        this.showMessage('فقط فایل‌های PDF و Excel پذیرفته می‌شوند.', 'error');
        event.target.value = '';
      }
    }
  }

  onFileDropZoneClick(): void {
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    if (fileInput) {
      fileInput.click();
    }
  }

  submitContribution() {
    if (!this.isFormValid()) {
      this.showMessage('لطفاً تمام فیلدهای ضروری را پر کنید.', 'error');
      return;
    }
    debugger
    this.isSubmitting = true;

    const formData = new FormData();
    formData.append('title', this.form.title);
    formData.append('content', this.form.content);
    formData.append('source', this.form.source);
    formData.append('meta_tags', this.form.tags);
    formData.append('author_name', this.form.author || '');
    formData.append('additional_references', this.form.additionalReferences || '');

    if (this.selectedFile) {
      formData.append('file', this.selectedFile);
    }

    this.contributeService.submitContribution(formData).subscribe({
      next: (response) => {
        this.isSubmitting = false;
        console.log(response);
        this.showMessage('مشارکت شما با موفقیت ثبت شد. متشکریم!', 'success');
        this.resetForm();
      },
      error: (error) => {
        this.isSubmitting = false;
        console.error('Contribution error:', error);
        this.showMessage('خطا در ثبت مشارکت. لطفاً دوباره تلاش کنید.', 'error');
      }
    });
  }

  uploadFile() {
    if (!this.selectedFile) {
      this.showMessage('لطفاً ابتدا فایل را انتخاب کنید.', 'error');
      return;
    }

    this.isSubmitting = true;
    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.contributeService.uploadFile(formData).subscribe({
      next: (response) => {
        this.isSubmitting = false;
        this.showMessage('فایل با موفقیت آپلود شد.', 'success');
        this.selectedFile = null;
        // Reset file input
        const fileInput = document.getElementById('fileInput') as HTMLInputElement;
        if (fileInput) fileInput.value = '';
      },
      error: (error) => {
        this.isSubmitting = false;
        console.error('File upload error:', error);
        this.showMessage('خطا در آپلود فایل. لطفاً دوباره تلاش کنید.', 'error');
      }
    });
  }

  public isFormValid(): boolean {
    return !!(this.form.title.trim() &&
      this.form.content.trim() &&
      this.form.category &&
      this.form.tags.trim() &&
      this.form.source.trim() &&
      this.form.author.trim() &&
      this.form.additionalReferences.trim());
  }

  private resetForm() {
    this.form = {
      title: '',
      content: '',
      category: '',
      tags: '',
      source: '',
      author: '',
      additionalReferences: ''
    };
    this.selectedFile = null;
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  }

  private showMessage(message: string, type: 'success' | 'error' = 'success') {
    this.snackBar.open(message, 'بستن', {
      duration: 5000,
      horizontalPosition: 'center',
      verticalPosition: 'top',
      panelClass: type === 'success' ? 'success-snackbar' : 'error-snackbar'
    });
  }

  getCategoryLabel(value: string): string {
    const category = this.categories.find(cat => cat.value === value);
    return category ? category.label : value;
  }
}