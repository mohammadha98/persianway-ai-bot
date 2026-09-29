import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  ApiResponse,
  TavilySearchSettings,
  TavilyTestResponse,
} from '../models/tavily-settings.model';

@Injectable({
  providedIn: 'root'
})
export class WebToolService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';
  // Lives on the same `config` router as the LLM/RAG settings pages.
  private tavilyBase = `${this.apiUrl}/api/config/tavily`;

  constructor(private http: HttpClient) { }

  getTavilySettings(): Observable<ApiResponse<TavilySearchSettings>> {
    // Same cache busting as the LLM/RAG pages: a proxy or CDN must never serve a
    // stale copy of settings an operator just changed.
    const headers = new HttpHeaders({
      accept: 'application/json',
      'Cache-Control': 'no-cache',
      Pragma: 'no-cache'
    });

    const params = new HttpParams().set('_t', Date.now().toString());

    return this.http.get<ApiResponse<TavilySearchSettings>>(this.tavilyBase, {
      headers,
      params
    });
  }

  /** Sends only the supplied keys; the backend keeps the fields that are missing. */
  updateTavilySettings(settings: Partial<TavilySearchSettings>): Observable<ApiResponse<TavilySearchSettings>> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
      accept: 'application/json'
    });

    return this.http.put<ApiResponse<TavilySearchSettings>>(this.tavilyBase, settings, {
      headers
    });
  }

  /** Runs one real search with the settings already saved on the server. */
  testTavilySearch(query: string): Observable<TavilyTestResponse> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
      accept: 'application/json'
    });

    return this.http.post<TavilyTestResponse>(`${this.tavilyBase}/test`, { query }, {
      headers
    });
  }

  /** Restores only the web search section, leaving the LLM/RAG settings untouched. */
  resetTavilySettings(): Observable<ApiResponse<TavilySearchSettings>> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
      accept: 'application/json'
    });

    return this.http.post<ApiResponse<TavilySearchSettings>>(`${this.tavilyBase}/reset`, {}, {
      headers
    });
  }
}
