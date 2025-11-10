import { Component, OnInit, OnDestroy, ViewEncapsulation } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatTableModule } from '@angular/material/table';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatPaginatorModule, PageEvent } from '@angular/material/paginator';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { ContributeService, KnowledgeListItem } from '../../services/contribute.service';
import { KnowledgeDetailsModalComponent } from '../../modals/knowledge-details-modal/knowledge-details-modal.component';
import { Subject } from 'rxjs';
import { debounceTime, distinctUntilChanged, takeUntil } from 'rxjs/operators';

// Interface for display in the table
interface KnowledgeItem extends KnowledgeListItem {
  file_type?: string;
  file_name?: string;
  meta_tags: string[];
}

interface CategoryOption {
  value: string;
  label: string;
}

@Component({
  selector: 'app-knowledge-list',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatIconModule,
    MatButtonModule,
    MatTableModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatPaginatorModule,
    MatTooltipModule,
    MatSnackBarModule,
    MatDialogModule
  ],
  templateUrl: './knowledge-list.component.html',
  styleUrl: './knowledge-list.component.scss',
})
export class KnowledgeListComponent implements OnInit, OnDestroy {
  // Table configuration
  displayedColumns: string[] = ['title', 'tags', 'author', 'synced', 'actions'];
  
  // Pagination
  pageSize = 10;
  pageSizeOptions = [5, 10, 25, 50];
  totalItems = 0;
  currentPage = 0;
  
  // Search and filter
  searchText = '';
  selectedCategory = '';
  searchTextChanged = new Subject<string>();
  private destroy$ = new Subject<void>();
  isSearching = false;
  
  // Knowledge data
  knowledgeItems: KnowledgeItem[] = [];
  filteredKnowledgeItems: KnowledgeItem[] = [];
  isLoading = false;
  
  // Categories
  categories: CategoryOption[] = [
    { value: 'general', label: 'عمومی' },
    { value: 'technical', label: 'فنی' },
    { value: 'business', label: 'کسب و کار' },
    { value: 'legal', label: 'حقوقی' },
    { value: 'health', label: 'سلامت' }
  ];
  
  constructor(
    private contributeService: ContributeService,
    private snackBar: MatSnackBar,
    private dialog: MatDialog
  ) { }
  
  ngOnInit(): void {
    this.loadKnowledgeData();
    
    // Setup search with debounce
    this.searchTextChanged.pipe(
      debounceTime(300),
      distinctUntilChanged(),
      takeUntil(this.destroy$)
    ).subscribe(() => {
      this.isSearching = true;
      this.applyFilters();
      this.isSearching = false;
    });
  }
  
  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
  
  onSearchInput(event: Event): void {
    const value = (event.target as HTMLInputElement).value;
    this.searchTextChanged.next(value);
  }
  
  loadKnowledgeData(): void {
    this.isLoading = true;
    this.contributeService.knowledgeList().subscribe({
      next: (data) => {
        // Map the API response to our KnowledgeItem interface
        this.knowledgeItems = data.map(item => ({
          ...item,
          // Ensure meta_tags is always an array
          meta_tags: item.meta_tags || []
        }));
        
        this.totalItems = this.knowledgeItems.length;
        this.applyFilters();
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error fetching knowledge list:', error);
        this.isLoading = false;
      }
    });
  }
  
  refresh(): void {
    this.currentPage = 0;
    this.loadKnowledgeData();
  }
  

  applyFilters(): void {
    let filtered = [...this.knowledgeItems];
    
  
    
    // Apply search filter
    if (this.searchText) {
      const searchLower = this.searchText.toLowerCase();
      filtered = filtered.filter(item => 
        item.title.toLowerCase().includes(searchLower) || 
        item.content.toLowerCase().includes(searchLower) || 
        (item.meta_tags && item.meta_tags.some(tag => tag.toLowerCase().includes(searchLower)))
      );
    }
    
    this.totalItems = filtered.length;
    
    // Apply pagination
    const startIndex = this.currentPage * this.pageSize;
    this.filteredKnowledgeItems = filtered.slice(startIndex, startIndex + this.pageSize);
  }
  
  onPageChange(event: PageEvent): void {
    this.currentPage = event.pageIndex;
    this.pageSize = event.pageSize;
    this.applyFilters();
  }
  
  getCategoryLabel(value: string): string {
    const category = this.categories.find(c => c.value === value);
    return category ? category.label : value;
  }
  
  viewKnowledge(item: KnowledgeItem): void {
    if (!item) {
      this.snackBar.open('خطا: اطلاعات سند یافت نشد', 'بستن', {
        duration: 3000,
        panelClass: ['error-snackbar']
      });
      return;
    }

    try {
      const dialogRef = this.dialog.open(KnowledgeDetailsModalComponent, {
        width: '800px',
        maxWidth: '95vw',
        maxHeight: '90vh',
        data: { knowledge: item },
        panelClass: 'knowledge-details-modal',
        autoFocus: true,
        restoreFocus: true,
        disableClose: false
      });

      // Optional: Handle dialog close events
      dialogRef.afterClosed().subscribe(result => {
        // Handle any cleanup or actions after modal closes
        console.log('Knowledge details modal closed');
      });

    } catch (error) {
      console.error('Error opening knowledge details modal:', error);
      this.snackBar.open('خطا در نمایش جزئیات سند', 'بستن', {
        duration: 3000,
        panelClass: ['error-snackbar']
      });
    }
  }
  
  editKnowledge(item: KnowledgeItem): void {
    console.log('Edit knowledge item:', item);
    // Implement edit functionality
  }
  
  deleteKnowledge(item: KnowledgeItem): void {
    console.log('Delete knowledge item:', item);
    
    if (confirm(`آیا از حذف "${item.title}" اطمینان دارید؟`)) {
      this.contributeService.removeKnowledge(item.hash_id).subscribe({
        next: (response) => {
          // Remove item from local array
          this.applyFilters();
          this.loadKnowledgeData();
          // Show success notification
          this.snackBar.open(
            `سند "${item.title}" با موفقیت حذف شد`,
            'بستن',
            {
              duration: 3000,
              horizontalPosition: 'center',
              verticalPosition: 'top',
              panelClass: ['success-snackbar']
            }
          );
        },
        error: (error) => {
          console.error('Error deleting knowledge item:', error);
          
          // Show error notification
          this.snackBar.open(
            `خطا در حذف سند "${item.title}". لطفاً دوباره تلاش کنید.`,
            'بستن',
            {
              duration: 5000,
              horizontalPosition: 'center',
              verticalPosition: 'top',
              panelClass: ['error-snackbar']
            }
          );
        }
      });
    }
  }
}
