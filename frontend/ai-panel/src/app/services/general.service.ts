import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { environment } from '../../environments/environment';

interface AppSettings {
  project_name: string;
  project_description: string;
  version: string;
  api_prefix: string;
  host: string;
  port: number;
  debug: boolean;
  allowed_hosts: string[];
}

interface AppConfigResponse {
  success: boolean;
  message: string;
  settings: AppSettings;
}

interface HealthResponse {
  status: string;
  version: string;
}

interface KnowledgeStatusResponse {
  status: string;
  document_count: number;
  pdf_document_count: number;
  excel_qa_count: number;
}

interface UpdateConfigRequest {
  app_settings: AppSettings;
}

interface UpdateConfigResponse {
  success: boolean;
  message: string;
  updated_sections: string[];
}


@Injectable({
  providedIn: 'root'
})
export class GeneralService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';
  
  private storageKey = 'persianway-settings';

  constructor(private http: HttpClient) {}

  getAppConfig(): Observable<AppConfigResponse> {
    return this.http.get<AppConfigResponse>(`${this.apiUrl}/api/config/app`);
  }
  
  checkHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.apiUrl}/health`);
  }

  getKnowledgeStatus(): Observable<KnowledgeStatusResponse> {
    return this.http.get<KnowledgeStatusResponse>(`${this.apiUrl}/api/knowledge/status`);
  }

  updateConfig(appSettings: AppSettings): Observable<UpdateConfigResponse> {
    const requestBody: UpdateConfigRequest = {
      app_settings: appSettings
    };
    return this.http.put<UpdateConfigResponse>(`${this.apiUrl}/api/config/`, requestBody);
  }
}