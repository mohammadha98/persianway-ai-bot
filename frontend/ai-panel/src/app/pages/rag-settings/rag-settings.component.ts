import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatDividerModule } from '@angular/material/divider';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { RagService } from '../../services/rag.service';

interface RagSettings {
  knowledge_base_confidence_threshold: number;
  qa_match_threshold: number;
  qa_priority_factor: number;
  human_referral_message: string;
  excel_qa_path: string;
  search_type: string;
  top_k_results: number;
  temperature: number;
  prompt_template: string;
  system_prompt: string;
}

@Component({
  selector: 'app-rag-settings',
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatIconModule,
    MatSlideToggleModule,
    MatDividerModule,
    MatSnackBarModule,
    MatExpansionModule,
    
  ],
  templateUrl: './rag-settings.component.html',
  styleUrl: './rag-settings.component.scss'
})
export class RagSettingsComponent implements OnInit {
  ragSettings: RagSettings = {
    knowledge_base_confidence_threshold: 0.7,
    qa_match_threshold: 0.6,
    qa_priority_factor: 1.5,
    human_referral_message: '',
    excel_qa_path: 'docs',
    search_type: 'similarity',
    top_k_results: 5,
    temperature: 0.1,
    prompt_template: '',
    system_prompt: ''
  };

  searchTypes = [
    { value: 'similarity', label: 'شباهت (Similarity)' },
    { value: 'mmr', label: 'حداکثر تنوع حاشیه‌ای (MMR)' },
    { value: 'similarity_score_threshold', label: 'آستانه امتیاز شباهت' }
  ];

  isLoading = false;
  isSaving = false;

  constructor(
    private ragService: RagService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.loadRagSettings();
  }

  loadRagSettings(): void {
    this.isLoading = true;
    this.ragService.getRagConfig().subscribe({
      next: (response) => {
        if (response.success) {
          this.ragSettings = { ...response.settings };
          this.snackBar.open('تنظیمات RAG با موفقیت بارگذاری شد', 'بستن', {
            duration: 3000,
            horizontalPosition: 'center',
            verticalPosition: 'top'
          });
        }
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error loading RAG settings:', error);
        this.snackBar.open('خطا در بارگذاری تنظیمات RAG', 'بستن', {
          duration: 5000,
          horizontalPosition: 'center',
          verticalPosition: 'top'
        });
        this.isLoading = false;
      }
    });
  }

  saveSettings(): void {
    this.isSaving = true;
    this.ragService.updateRagConfig(this.ragSettings).subscribe({
      next: (response) => {
        if (response.success) {
          this.snackBar.open('تنظیمات RAG با موفقیت ذخیره شد', 'بستن', {
            duration: 3000,
            horizontalPosition: 'center',
            verticalPosition: 'top'
          });
        }
        this.isSaving = false;
      },
      error: (error) => {
        console.error('Error saving RAG settings:', error);
        this.snackBar.open('خطا در ذخیره تنظیمات RAG', 'بستن', {
          duration: 5000,
          horizontalPosition: 'center',
          verticalPosition: 'top'
        });
        this.isSaving = false;
      }
    });
  }

  resetSettings(): void {
    this.ragSettings = {
      knowledge_base_confidence_threshold: 0.7,
      qa_match_threshold: 0.6,
      qa_priority_factor: 1.5,
      human_referral_message: 'متأسفانه، اطلاعات کافی در پایگاه دانش برای پاسخ به این سؤال وجود ندارد. سؤال شما برای بررسی بیشتر توسط کارشناسان ما ثبت شده است.',
      excel_qa_path: 'docs',
      search_type: 'similarity',
      top_k_results: 5,
      temperature: 0.1,
      prompt_template: 'با استفاده از اطلاعات زیر، به سوال پاسخ دهید. اگر اطلاعات کافی نیست، صادقانه بگویید که نمی‌دانید.\n\nاطلاعات مرجع:\n{context}\n\nسوال: {question}\n\nپاسخ:',
      system_prompt: 'شما یک دستیار هوشمند تخصصی در حوزه سلامت، زیبایی و کشاورزی هستید. وظیفه شما ارائه پاسخ‌های مفید و دقیق بر اساس دانش تخصصی شما است.'
    };
    
    this.snackBar.open('تنظیمات به حالت پیش‌فرض بازگردانده شد', 'بستن', {
      duration: 3000,
      horizontalPosition: 'center',
      verticalPosition: 'top'
    });
  }
}
