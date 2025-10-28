import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface ContributionResponse {
  message: string;
  status: string;
  id?: string;
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
  synced?:boolean
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

  constructor(private http: HttpClient) {}

  submitContribution(formData: FormData): Observable<ContributionResponse> {
    const url = `${this.apiUrl}/api/knowledge/contribute`;
    return this.http.post<ContributionResponse>(url, formData);
  }

  uploadFile(formData: FormData): Observable<FileUploadResponse> {
    const url = `${this.apiUrl}/api/upload`;
    return this.http.post<FileUploadResponse>(url, formData);
  }

  // Method to get contribution categories if needed
  getCategories(): Observable<any> {
    const url = `${this.apiUrl}/api/categories`;
    return this.http.get(url);
  }

  // Method to get user contributions if needed
  getUserContributions(): Observable<any> {
    const url = `${this.apiUrl}/api/contributions`;
    return this.http.get(url);
  }


  knowledgeList(): Observable<KnowledgeListItem[]> {
    const url = `${this.apiUrl}/api/knowledge/knowledge-list`;
    return this.http.get<KnowledgeListItem[]>(url);
  }

  removeKnowledge(hashId: string): Observable<KnowledgeRemovalResponse> {
    const url = `${this.apiUrl}/api/knowledge/remove/${hashId}`;
    return this.http.delete<KnowledgeRemovalResponse>(url);
  }

  
}