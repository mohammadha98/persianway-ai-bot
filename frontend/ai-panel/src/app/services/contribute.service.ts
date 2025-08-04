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
}