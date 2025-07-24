import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSnackBarModule, MatSnackBar } from '@angular/material/snack-bar';
import { MatDividerModule } from '@angular/material/divider';
import { SettingsService } from '../../services/settings.service';

interface UserSettings {
  language: string;
  theme: string;
  notifications: boolean;
  autoSave: boolean;
  responseLength: string;
  apiTimeout: number;
  maxTokens: number;
}

interface SystemInfo {
  version: string;
  lastUpdate: string;
  totalKnowledge: number;
  activeUsers: number;
  systemStatus: string;
}

@Component({
  selector: 'app-settings',
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
    MatSlideToggleModule,
    MatSnackBarModule,
    MatDividerModule
  ],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.scss'
})
export class SettingsComponent implements OnInit {
  settings: UserSettings = {
    language: 'fa',
    theme: 'light',
    notifications: true,
    autoSave: true,
    responseLength: 'medium',
    apiTimeout: 30,
    maxTokens: 2048
  };

  systemInfo: SystemInfo = {
    version: '1.0.0',
    lastUpdate: '1403/05/03',
    totalKnowledge: 0,
    activeUsers: 0,
    systemStatus: 'فعال'
  };

  languages = [
    { value: 'fa', label: 'فارسی' },
    { value: 'en', label: 'English' }
  ];

  themes = [
    { value: 'light', label: 'روشن' },
    { value: 'dark', label: 'تیره' },
    { value: 'auto', label: 'خودکار' }
  ];

  responseLengths = [
    { value: 'short', label: 'کوتاه' },
    { value: 'medium', label: 'متوسط' },
    { value: 'long', label: 'بلند' }
  ];

  isLoading = false;
  isSaving = false;

  constructor(
    private settingsService: SettingsService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit() {
    this.loadSettings();
    this.loadSystemInfo();
  }

  loadSettings() {
    this.isLoading = true;
    this.settingsService.getSettings().subscribe({
      next: (settings) => {
        this.settings = { ...this.settings, ...settings };
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error loading settings:', error);
        this.isLoading = false;
        // Use default settings if loading fails
      }
    });
  }

  loadSystemInfo() {
    this.settingsService.getSystemInfo().subscribe({
      next: (info) => {
        this.systemInfo = { ...this.systemInfo, ...info };
      },
      error: (error) => {
        console.error('Error loading system info:', error);
        // Use default system info if loading fails
      }
    });
  }

  saveSettings() {
    this.isSaving = true;
    this.settingsService.saveSettings(this.settings).subscribe({
      next: (response) => {
        this.isSaving = false;
        this.showMessage('تنظیمات با موفقیت ذخیره شد.', 'success');
      },
      error: (error) => {
        this.isSaving = false;
        console.error('Error saving settings:', error);
        this.showMessage('خطا در ذخیره تنظیمات. لطفاً دوباره تلاش کنید.', 'error');
      }
    });
  }

  resetSettings() {
    this.settings = {
      language: 'fa',
      theme: 'light',
      notifications: true,
      autoSave: true,
      responseLength: 'medium',
      apiTimeout: 30,
      maxTokens: 2048
    };
    this.showMessage('تنظیمات به حالت پیش‌فرض بازگردانده شد.', 'success');
  }

  exportSettings() {
    const dataStr = JSON.stringify(this.settings, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'persianway-settings.json';
    link.click();
    URL.revokeObjectURL(url);
    this.showMessage('تنظیمات صادر شد.', 'success');
  }

  importSettings(event: any) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const importedSettings = JSON.parse(e.target?.result as string);
          this.settings = { ...this.settings, ...importedSettings };
          this.showMessage('تنظیمات وارد شد.', 'success');
        } catch (error) {
          this.showMessage('خطا در وارد کردن فایل. فرمت فایل صحیح نیست.', 'error');
        }
      };
      reader.readAsText(file);
    }
  }

  clearCache() {
    this.settingsService.clearCache().subscribe({
      next: (response) => {
        this.showMessage('کش سیستم پاک شد.', 'success');
      },
      error: (error) => {
        console.error('Error clearing cache:', error);
        this.showMessage('خطا در پاک کردن کش.', 'error');
      }
    });
  }

  private showMessage(message: string, type: 'success' | 'error' = 'success') {
    this.snackBar.open(message, 'بستن', {
      duration: 5000,
      horizontalPosition: 'center',
      verticalPosition: 'top',
      panelClass: type === 'success' ? 'success-snackbar' : 'error-snackbar'
    });
  }

  getLanguageLabel(value: string): string {
    const language = this.languages.find(lang => lang.value === value);
    return language ? language.label : value;
  }

  getThemeLabel(value: string): string {
    const theme = this.themes.find(t => t.value === value);
    return theme ? theme.label : value;
  }

  getResponseLengthLabel(value: string): string {
    const length = this.responseLengths.find(l => l.value === value);
    return length ? length.label : value;
  }
}