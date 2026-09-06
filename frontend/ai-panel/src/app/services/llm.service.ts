import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface LlmSettings {
  preferred_api_provider: string;
  default_model: string;
  available_models: string[];
  temperature: number;
  top_p: number;
  max_tokens: number;
  openai_api_key: string;
  openrouter_api_key: string;
  openrouter_api_base: string;
  openai_embedding_model: string;
}

export interface LlmSettingsResponse {
  success: boolean;
  message: string;
  settings: LlmSettings;
}

export interface LlmSettingsUpdateRequest {
  preferred_api_provider?: string;
  default_model?: string;
  available_models?: string[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  openai_api_key?: string;
  openrouter_api_key?: string;
  openrouter_api_base?: string;
  openai_embedding_model?: string;
}

@Injectable({
  providedIn: 'root',
})
export class LlmService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  getLlmSettings(): Observable<LlmSettingsResponse> {
    const headers = new HttpHeaders({
      accept: 'application/json',
      'Cache-Control': 'no-cache',
      Pragma: 'no-cache',
    });

    const params = new HttpParams().set('_t', Date.now().toString());

    return this.http.get<LlmSettingsResponse>(`${this.apiUrl}/api/config/llm`, {
      headers,
      params,
    });
  }

  updateLlmSettings(
    settings: LlmSettingsUpdateRequest,
  ): Observable<LlmSettingsResponse> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
      accept: 'application/json',
    });
    return this.http.put<LlmSettingsResponse>(
      `${this.apiUrl}/api/config/llm`,
      settings,
      { headers },
    );
  }
}
