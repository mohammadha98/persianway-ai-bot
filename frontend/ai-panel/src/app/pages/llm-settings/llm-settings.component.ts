import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { trigger, state, style, transition, animate } from '@angular/animations';
import { LlmService, LlmSettings, LlmSettingsUpdateRequest } from '../../services/llm.service';
import { ModelsService, OpenRouterModel, PaginatedModelsResponse } from '../../services/models.service';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { ModelDetailsModalComponent } from '../../modals/model-details-modal/model-details-modal.component';

@Component({
  selector: 'app-llm-settings',
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
    MatProgressSpinnerModule,
    MatChipsModule,
    MatTooltipModule,
    MatDialogModule,
  ],
  templateUrl: './llm-settings.component.html',
  styleUrl: './llm-settings.component.scss',
  animations: [
    trigger('expandCollapse', [
      state('collapsed', style({
        height: '0px',
        opacity: 0,
        overflow: 'hidden'
      })),
      state('expanded', style({
        height: '*',
        opacity: 1,
        overflow: 'visible'
      })),
      transition('collapsed <=> expanded', [
        animate('300ms cubic-bezier(0.4, 0.0, 0.2, 1)')
      ])
    ])
  ]
})
export class LlmSettingsComponent implements OnInit {
  llmSettings: LlmSettings | null = null;
  availableModels: OpenRouterModel[] = [];
  paginatedResponse: PaginatedModelsResponse | null = null;
  isLoading = false;
  isSaving = false;
  isLoadingModels = false;
  
  // Pagination properties
  currentPage = 1;
  pageSize = 12;
  searchTerm = '';
  totalPages = 0;
  
  // Expandable section properties
  isModelsExpanded = true;

  apiProviders = [
    { value: 'openai', label: 'OpenAI' },
    { value: 'openrouter', label: 'OpenRouter' },
  ];

  constructor(
    private llmService: LlmService,
    private modelsService: ModelsService,
    private snackBar: MatSnackBar,
    private dialog: MatDialog
  ) {}

  ngOnInit() {
    this.loadLlmSettings();
    this.loadAvailableModels();
  }

  loadLlmSettings() {
    this.isLoading = true;
    this.llmService.getLlmSettings().subscribe({
      next: (response) => {
        if (response.success) {
          this.llmSettings = response.settings;
        } else {
          this.showMessage('خطا در بارگذاری تنظیمات', 'error');
        }
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error loading LLM settings:', error);
        this.showMessage('خطا در اتصال به سرور', 'error');
        this.isLoading = false;
      }
    });
  }

  saveSettings() {
    if (!this.llmSettings) return;

    this.isSaving = true;
    const updateRequest: LlmSettingsUpdateRequest = {
      preferred_api_provider: this.llmSettings.preferred_api_provider,
      default_model: this.llmSettings.default_model,
      temperature: this.llmSettings.temperature,
      top_p: this.llmSettings.top_p,
      max_tokens: this.llmSettings.max_tokens,
      openai_api_key: this.llmSettings.openai_api_key,
      openrouter_api_key: this.llmSettings.openrouter_api_key,
      openrouter_api_base: this.llmSettings.openrouter_api_base,
      openai_embedding_model: this.llmSettings.openai_embedding_model
    };

    this.llmService.updateLlmSettings(updateRequest).subscribe({
      next: (response) => {
        if (response.success) {
          this.llmSettings = response.settings;
          this.showMessage('تنظیمات با موفقیت ذخیره شد', 'success');
        } else {
          this.showMessage('خطا در ذخیره تنظیمات', 'error');
        }
        this.isSaving = false;
      },
      error: (error) => {
        console.error('Error saving LLM settings:', error);
        this.showMessage('خطا در ذخیره تنظیمات', 'error');
        this.isSaving = false;
      }
    });
  }

  resetSettings() {
    this.loadLlmSettings();
    this.showMessage('تنظیمات بازیابی شد', 'success');
  }

  loadAvailableModels() {
    this.isLoadingModels = true;
    this.modelsService.getPaginatedModels(this.currentPage, this.pageSize, this.searchTerm).subscribe({
      next: (response) => {
        this.paginatedResponse = response;
        this.availableModels = response.data;
        this.totalPages = response.totalPages;
        this.isLoadingModels = false;
      },
      error: (error) => {
        console.error('Error loading available models:', error);
        this.showMessage('خطا در بارگذاری مدل‌ها', 'error');
        this.isLoadingModels = false;
      }
    });
  }

  onModelClick(model: OpenRouterModel): void {
    const dialogRef = this.dialog.open(ModelDetailsModalComponent, {
      data: model,
      width: '800px',
      maxWidth: '90vw',
      maxHeight: '90vh',
      panelClass: 'model-details-dialog'
    });
  }

  trackByModelId(index: number, model: OpenRouterModel): string {
    return model.id;
  }

  formatContextLength(contextLength: number): string {
    if (contextLength >= 1000000) {
      return `${(contextLength / 1000000).toFixed(1)}M`;
    } else if (contextLength >= 1000) {
      return `${(contextLength / 1000).toFixed(0)}K`;
    }
    return contextLength.toString();
  }

  formatPrice(price: string): string {
    const numPrice = parseFloat(price);
    if (numPrice === 0) {
      return 'رایگان';
    }
    if (numPrice < 0.001) {
      return `$${(numPrice * 1000000).toFixed(2)}/1M`;
    }
    return `$${numPrice.toFixed(4)}/1K`;
  }

  getProviderName(modelId: string): string {
    const providers: { [key: string]: string } = {
      'openai': 'OpenAI',
      'anthropic': 'Anthropic',
      'google': 'Google',
      'meta': 'Meta',
      'mistral': 'Mistral',
      'cohere': 'Cohere',
      'perplexity': 'Perplexity',
      'huggingface': 'Hugging Face',
      'together': 'Together AI',
      'fireworks': 'Fireworks',
      'deepseek': 'DeepSeek',
      'qwen': 'Qwen',
      'yi': 'Yi',
      'zhipu': 'Zhipu'
    };

    for (const [key, name] of Object.entries(providers)) {
      if (modelId.toLowerCase().includes(key)) {
        return name;
      }
    }
    
    // Extract provider from model ID format (provider/model)
    const parts = modelId.split('/');
    if (parts.length > 1) {
      return parts[0].charAt(0).toUpperCase() + parts[0].slice(1);
    }
    
    return 'سایر';
  }

  onSearchChange() {
    this.currentPage = 1; // Reset to first page when searching
    this.loadAvailableModels();
  }

  onPageChange(page: number) {
    if (page >= 1 && page <= this.totalPages) {
      this.currentPage = page;
      this.loadAvailableModels();
    }
  }

  onPageSizeChange(newPageSize: number) {
    this.pageSize = newPageSize;
    this.currentPage = 1; // Reset to first page
    this.loadAvailableModels();
  }

  clearCache() {
    this.modelsService.clearCache();
    this.currentPage = 1;
    this.loadAvailableModels();
    this.showMessage('کش مدل‌ها پاک شد', 'success');
  }

  getPageNumbers(): number[] {
    const pages: number[] = [];
    const maxVisiblePages = 5;
    const halfVisible = Math.floor(maxVisiblePages / 2);
    
    let startPage = Math.max(1, this.currentPage - halfVisible);
    let endPage = Math.min(this.totalPages, startPage + maxVisiblePages - 1);
    
    // Adjust start page if we're near the end
    if (endPage - startPage < maxVisiblePages - 1) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
      pages.push(i);
    }
    
    return pages;
  }
  
  toggleModelsExpanded() {
    this.isModelsExpanded = !this.isModelsExpanded;
  }

  private showMessage(message: string, type: 'success' | 'error') {
    this.snackBar.open(message, 'بستن', {
      duration: 3000,
      panelClass: type === 'success' ? 'success-snackbar' : 'error-snackbar',
      horizontalPosition: 'center',
      verticalPosition: 'top'
    });
  }
}
