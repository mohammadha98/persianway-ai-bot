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
import { GeneralService } from '../../services/general.service';
import { forkJoin } from 'rxjs';

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
  projectName?: string;
  apiStatus?: string;
}

interface AppConfigSettings {
  project_name: string;
  project_description: string;
  version: string;
  api_prefix: string;
  host: string;
  port: number;
  debug: boolean;
  allowed_hosts: string[];
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
    systemStatus: 'فعال',
    projectName: 'Persian Way AI Bot',
    apiStatus: 'در حال بررسی...'
  };

  appSettings: AppConfigSettings = {
    project_name: '',
    project_description: '',
    version: '',
    api_prefix: '',
    host: '',
    port: 8000,
    debug: false,
    allowed_hosts: []
  };

  allowedHostsString: string = '';
  isUpdatingConfig = false;

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
    private generalService: GeneralService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit() {
    this.loadSettings();
    this.loadSystemInfo();
  }

  loadSettings() {
    this.isLoading = true;
    try {
      const savedSettings = localStorage.getItem('persianway-settings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        this.settings = { ...this.settings, ...parsedSettings };
      }
    } catch (error) {
      console.error('Error loading settings from localStorage:', error);
    }
    this.isLoading = false;
  }

  loadSystemInfo() {
    // Use forkJoin to make parallel API calls
    forkJoin({
      appConfig: this.generalService.getAppConfig(),
      health: this.generalService.checkHealth(),
      knowledgeStatus: this.generalService.getKnowledgeStatus()
    }).subscribe({
      next: (responses) => {
         // Update system info with real data from APIs
         this.systemInfo = {
           version: responses.appConfig.settings.version,
           lastUpdate: new Date().toLocaleDateString('fa-IR'),
           totalKnowledge: responses.knowledgeStatus.document_count + responses.knowledgeStatus.pdf_document_count + responses.knowledgeStatus.excel_qa_count,
           activeUsers: 1, // This would need to come from another API
           systemStatus: responses.health.status === 'healthy' ? 'فعال' : 'غیرفعال',
           projectName: responses.appConfig.settings.project_name,
           apiStatus: 'متصل'
         };
         
         // Load app settings for editing
         this.appSettings = { ...responses.appConfig.settings };
         this.allowedHostsString = this.appSettings.allowed_hosts.join(', ');
      },
      error: (error) => {
        console.error('Error loading system info:', error);
        // Use default system info if loading fails
        this.showMessage('خطا در بارگذاری اطلاعات سیستم', 'error');
      }
    });
  }

  saveSettings() {
    this.isSaving = true;
    try {
      localStorage.setItem('persianway-settings', JSON.stringify(this.settings));
      this.isSaving = false;
      this.showMessage('تنظیمات با موفقیت ذخیره شد.', 'success');
    } catch (error) {
      this.isSaving = false;
      console.error('Error saving settings:', error);
      this.showMessage('خطا در ذخیره تنظیمات. لطفاً دوباره تلاش کنید.', 'error');
    }
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
    try {
      localStorage.clear();
      sessionStorage.clear();
      this.showMessage('کش سیستم پاک شد.', 'success');
    } catch (error) {
      console.error('Error clearing cache:', error);
      this.showMessage('خطا در پاک کردن کش.', 'error');
    }
  }

  updateAppConfig() {
    this.isUpdatingConfig = true;
    
    // Parse allowed hosts string back to array
    const allowedHostsArray = this.allowedHostsString
      .split(',')
      .map(host => host.trim())
      .filter(host => host.length > 0);
    
    const updatedSettings = {
      ...this.appSettings,
      allowed_hosts: allowedHostsArray
    };
    
    this.generalService.updateConfig(updatedSettings).subscribe({
      next: (response) => {
        this.isUpdatingConfig = false;
        this.showMessage('تنظیمات برنامه با موفقیت بروزرسانی شد.', 'success');
        // Reload system info to reflect changes
        this.loadSystemInfo();
      },
      error: (error) => {
        this.isUpdatingConfig = false;
        console.error('Error updating app config:', error);
        this.showMessage('خطا در بروزرسانی تنظیمات برنامه.', 'error');
      }
    });
  }

  resetAppConfig() {
    // Reset to current loaded values
    this.loadSystemInfo();
    this.showMessage('تنظیمات برنامه به حالت اولیه بازگردانده شد.', 'success');
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