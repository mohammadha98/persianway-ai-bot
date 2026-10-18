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
  // OpenRouter credit alerts (app/services/credit_monitor.py). The threshold is
  // USD: at or below it, the chat page shows a top-up notice after every answer.
  openrouter_credit_alert_enabled: boolean;
  openrouter_credit_alert_threshold_usd: number;
  // Optional management key: only it may read the account balance
  // (GET /api/v1/credits); the inference key gets 403 there.
  openrouter_management_key: string;
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
  openrouter_credit_alert_enabled?: boolean;
  openrouter_credit_alert_threshold_usd?: number;
  openrouter_management_key?: string;
}

/**
 * Remaining OpenRouter credit, as answered by
 * `GET /api/config/llm/credits` (app/services/credit_monitor.py).
 */
export interface OpenRouterCredits {
  provider: string;
  /** False when OpenRouter is not the active provider, or alerts are off. */
  applicable: boolean;
  /** null = the balance cannot be read (no management key, no per-key limit). */
  balance_usd: number | null;
  threshold_usd: number;
  is_low: boolean;
  is_exhausted: boolean;
  source: 'account' | 'key' | 'provider_402' | 'unavailable';
  checked_at: string;
  message: string | null;
  topup_url: string;
  error: string | null;
}

export interface OpenRouterCreditsResponse {
  success: boolean;
  message: string;
  credits: OpenRouterCredits | null;
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

  /**
   * Read the remaining OpenRouter credit.
   *
   * `force` bypasses the backend's 60s cache: used by the settings page's
   * refresh button and right after an answer failed with HTTP 402.
   */
  getOpenRouterCredits(force = false): Observable<OpenRouterCreditsResponse> {
    const headers = new HttpHeaders({
      accept: 'application/json',
      'Cache-Control': 'no-cache',
      Pragma: 'no-cache',
    });

    const params = new HttpParams()
      .set('force', String(force))
      .set('_t', Date.now().toString());

    return this.http.get<OpenRouterCreditsResponse>(
      `${this.apiUrl}/api/config/llm/credits`,
      { headers, params },
    );
  }
}
