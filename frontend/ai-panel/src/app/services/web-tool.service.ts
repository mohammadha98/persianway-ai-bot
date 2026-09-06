import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { ApiResponse, TavilySearchSettings } from '../models/tavily-settings.model';

@Injectable({
  providedIn: 'root'
})
export class WebToolService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';
  private configBase = `${this.apiUrl}/config`;
  private tavilyBase = `${this.configBase}/tavily`;

  constructor(private http: HttpClient) { }

  getTavilySettings(): Observable<ApiResponse<TavilySearchSettings>> {
    return this.http.get<ApiResponse<TavilySearchSettings>>(this.tavilyBase);
  }

  updateTavilySettings(settings: Partial<TavilySearchSettings>): Observable<ApiResponse<TavilySearchSettings>> {
    return this.http.put<ApiResponse<TavilySearchSettings>>(this.tavilyBase, settings);
  }
}
