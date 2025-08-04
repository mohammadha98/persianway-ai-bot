import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { environment } from '../../environments/environment';

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

interface RagConfigResponse {
  success: boolean;
  message: string;
  settings: RagSettings;
}

@Injectable({
  providedIn: 'root'
})
export class RagService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';
  
  constructor(private http: HttpClient) { }

  getRagConfig(): Observable<RagConfigResponse> {
    const headers = new HttpHeaders({
      'accept': 'application/json'
    });
    
    return this.http.get<RagConfigResponse>(`${this.apiUrl}/api/config/rag`, { headers });
  }

  updateRagConfig(settings: RagSettings): Observable<RagConfigResponse> {
    const headers = new HttpHeaders({
      'accept': 'application/json',
      'Content-Type': 'application/json'
    });
    
    return this.http.put<RagConfigResponse>(`${this.apiUrl}/api/config/rag`, settings, { headers });
  }
}
