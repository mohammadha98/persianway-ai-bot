import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { ChatService, ConversationSearchRequest, ConversationListResponse, ConversationResponse } from '../../services/chat.service';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatPaginatorModule, PageEvent } from '@angular/material/paginator';
import { MatDialog } from '@angular/material/dialog';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ConversationDetailsModalComponent } from '../../modals/conversation-details-modal/conversation-details-modal.component';

@Component({
  selector: 'app-conversation-search',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatPaginatorModule
  ],
  templateUrl: './conversation-search.component.html',
  styleUrls: ['./conversation-search.component.scss']
})
export class ConversationSearchComponent implements OnInit {
  searchForm: FormGroup;
  conversations: ConversationResponse[] = [];
  loading = false;
  totalCount = 0;
  currentPage = 1;
  pageSize = 20;
  hasNext = false;

  knowledgeSources = [
    { value: '', label: 'همه منابع' },
    { value: 'knowledge_base', label: 'پایگاه دانش' },
    { value: 'general_knowledge', label: 'دانش عمومی' },
    { value: 'none', label: 'بدون منبع' }
  ];

  constructor(
    private fb: FormBuilder,
    private chatService: ChatService,
    private snackBar: MatSnackBar,
    private dialog: MatDialog
  ) {
    this.searchForm = this.fb.group({
      user_id: [''],
      start_date: [''],
      end_date: [''],
      search_text: [''],
      knowledge_source: [''],
      requires_human_referral: [null],
      min_confidence: [null],
      max_confidence: [null]
    });
  }

  ngOnInit(): void {
    this.searchConversations();
  }

  onSearch(): void {
    this.currentPage = 1;
    this.searchConversations();
  }

  onClear(): void {
    this.searchForm.reset();
    this.conversations = [];
    this.totalCount = 0;
    this.currentPage = 1;
    this.hasNext = false;
  }

  onPageChange(event: PageEvent): void {
    this.currentPage = event.pageIndex + 1;
    this.pageSize = event.pageSize;
    this.searchConversations();
  }

  searchConversations(): void {
    this.loading = true;
    const formValue = this.searchForm.value;
    
    // Prepare search criteria
    const searchCriteria: ConversationSearchRequest = {
      limit: this.pageSize,
      skip: (this.currentPage - 1) * this.pageSize
    };

    // Add non-empty form values to search criteria
    if (formValue.user_id?.trim()) {
      searchCriteria.user_id = formValue.user_id.trim();
    }
    if (formValue.start_date) {
      searchCriteria.start_date = new Date(formValue.start_date).toISOString();
    }
    if (formValue.end_date) {
      searchCriteria.end_date = new Date(formValue.end_date).toISOString();
    }
    if (formValue.search_text?.trim()) {
      searchCriteria.search_text = formValue.search_text.trim();
    }
    if (formValue.knowledge_source) {
      searchCriteria.knowledge_source = formValue.knowledge_source;
    }
    if (formValue.requires_human_referral !== null) {
      searchCriteria.requires_human_referral = formValue.requires_human_referral;
    }
    if (formValue.min_confidence !== null && formValue.min_confidence >= 0) {
      searchCriteria.min_confidence = formValue.min_confidence;
    }
    if (formValue.max_confidence !== null && formValue.max_confidence >= 0) {
      searchCriteria.max_confidence = formValue.max_confidence;
    }

    this.chatService.searchConversations(searchCriteria).subscribe({
      next: (response: ConversationListResponse) => {
        this.conversations = response.conversations;
        this.totalCount = response.total_count;
        this.hasNext = response.has_next;
        this.loading = false;
      },
      error: (error) => {
        console.error('Error searching conversations:', error);
        this.snackBar.open('خطا در جستجوی مکالمات', 'بستن', {
          duration: 3000,
          horizontalPosition: 'center',
          verticalPosition: 'top'
        });
        this.loading = false;
      }
    });
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString('fa-IR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  getKnowledgeSourceLabel(source: string): string {
    const sourceObj = this.knowledgeSources.find(s => s.value === source);
    return sourceObj ? sourceObj.label : source;
  }

  getTotalPages(): number {
    return Math.ceil(this.totalCount / this.pageSize);
  }

  getPageNumbers(): number[] {
    const totalPages = this.getTotalPages();
    const pages: number[] = [];
    const maxVisiblePages = 5;
    
    let startPage = Math.max(1, this.currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    
    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
      pages.push(i);
    }
    
    return pages;
  }

  openConversationDetails(conversation: ConversationResponse): void {
    this.dialog.open(ConversationDetailsModalComponent, {
      width: '80%',
      data: { conversation }
    });
  }
}