import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError, timer } from 'rxjs';
import { catchError, switchMap, takeUntil } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export type TaskStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
  progress: number;
  error?: string;
  metadata?: any;
  created_at: string;
  updated_at: string;
  completed_at?: string;
}

export interface KnowledgeContributionItem {
  id: string;
  title: string;
  submitted_at: string;
  meta_tags: string[];
  content?: string;
  source: string;
  author_name?: string;
  additional_references?: string;
  file_processed?: boolean;
  file_type?: string;
  file_name?: string;
  qa_count?: number;
  is_public: boolean;
  task_id?: string;
  status?: string;
}

export interface KnowledgeContributionResponse {
  success: boolean;
  contribution?: KnowledgeContributionItem;
  message?: string;
}

export interface ContributionResponse {
  message: string;
  status: string;
  id?: string;
  task_id?: string;
}

export interface FileUploadResponse {
  message: string;
  status: string;
  filename?: string;
}

export interface KnowledgeListItem{
  hash_id: string;
  title: string;
  content: string;
  meta_tags: string[];
  author_name: string;
  additionalReferences?: string;
  submission_timestamp: string;
  file_type?: string;
  file_name?: string;
  synced?:boolean;
  task_id?: string;
  task_status?: TaskStatus;
}

export interface PaginatedKnowledgeListResponse{
  items: KnowledgeListItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface KnowledgeRemovalResponse {
  success: boolean;
  message: string;
  hash_id: string;
  removed_count: number;
}

@Injectable({
  providedIn: 'root'
})
export class ContributeService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  submitContribution(formData: FormData): Observable<ContributionResponse> {
    const url = `${this.apiUrl}/api/knowledge/contribute`;
    return this.http.post<ContributionResponse>(url, formData).pipe(
      catchError(this.handleError)
    );
  }

  getTaskStatus(taskId: string): Observable<TaskStatusResponse> {
    const url = `${this.apiUrl}/api/knowledge/task/${taskId}`;
    return this.http.get<TaskStatusResponse>(url).pipe(
      catchError(this.handleError)
    );
  }

  pollTaskStatus(taskId: string, pollIntervalMs = 3000, timeoutMs = 900000): Observable<TaskStatusResponse> {
    const endTime = Date.now() + timeoutMs;
    
    return timer(0, pollIntervalMs).pipe(
      switchMap(() => this.getTaskStatus(taskId)),
      takeUntil(timer(timeoutMs))
    );
  }

  uploadFile(formData: FormData): Observable<FileUploadResponse> {
    const url = `${this.apiUrl}/api/upload`;
    return this.http.post<FileUploadResponse>(url, formData).pipe(
      catchError(this.handleError)
    );
  }

  // Method to get contribution categories if needed
  getCategories(): Observable<any> {
    const url = `${this.apiUrl}/api/categories`;
    return this.http.get(url).pipe(
      catchError(this.handleError)
    );
  }

  // Method to get user contributions if needed
  getUserContributions(): Observable<any> {
    const url = `${this.apiUrl}/api/contributions`;
    return this.http.get(url).pipe(
      catchError(this.handleError)
    );
  }

  knowledgeList(page: number = 1, pageSize: number = 10): Observable<PaginatedKnowledgeListResponse> {
    const url = `${this.apiUrl}/api/knowledge/knowledge-list?page=${page}&page_size=${pageSize}`;
    return this.http.get<PaginatedKnowledgeListResponse>(url).pipe(
      catchError(this.handleError)
    );
  }

  removeKnowledge(hashId: string): Observable<KnowledgeRemovalResponse> {
    const url = `${this.apiUrl}/api/knowledge/remove/${hashId}`;
    return this.http.delete<KnowledgeRemovalResponse>(url).pipe(
      catchError(this.handleError)
    );
  }

  private handleError(error: HttpErrorResponse) {
    let errorMessage = 'An unknown error occurred!';
    if (error.error instanceof ErrorEvent) {
      // Client-side or network error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Backend error
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }
    console.error(errorMessage);
    return throwError(() => new Error(errorMessage));
  }
}