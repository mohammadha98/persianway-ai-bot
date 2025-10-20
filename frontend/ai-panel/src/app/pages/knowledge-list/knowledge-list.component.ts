import { Component, OnInit, OnDestroy } from '@angular/core';
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
import { ContributeService, KnowledgeListItem } from '../../services/contribute.service';
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
    MatTooltipModule
  ],
  templateUrl: './knowledge-list.component.html',
  styleUrl: './knowledge-list.component.scss'
})
export class KnowledgeListComponent implements OnInit, OnDestroy {
  // Table configuration
  displayedColumns: string[] = ['title', 'tags', 'author', 'actions'];
  
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
  
  constructor(private contributeService: ContributeService) { }
  
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
    console.log('View knowledge item:', item);
    // Implement view functionality
  }
  
  editKnowledge(item: KnowledgeItem): void {
    console.log('Edit knowledge item:', item);
    // Implement edit functionality
  }
  
  deleteKnowledge(item: KnowledgeItem): void {
    console.log('Delete knowledge item:', item);
    // Implement delete functionality
    if (confirm(`آیا از حذف "${item.title}" اطمینان دارید؟`)) {
      // In a real implementation, you would call a service method to delete from the API
      this.knowledgeItems = this.knowledgeItems.filter(i => i.hash_id !== item.hash_id);
      this.applyFilters();
    }
  }
}
