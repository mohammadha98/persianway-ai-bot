import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { environment } from '../../environments/environment';

export interface UserSettings {
  language: string;
  theme: string;
  notifications: boolean;
  autoSave: boolean;
  responseLength: string;
  apiTimeout: number;
  maxTokens: number;
}

export interface SystemInfo {
  version: string;
  lastUpdate: string;
  totalKnowledge: number;
  activeUsers: number;
  systemStatus: string;
}

export interface SettingsResponse {
  message: string;
  status: string;
}

@Injectable({
  providedIn: 'root'
})
export class SettingsService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';
  
  private storageKey = 'persianway-settings';

  constructor(private http: HttpClient) {}

  getSettings(): Observable<UserSettings> {
    // Try to get settings from localStorage first
    const savedSettings = localStorage.getItem(this.storageKey);
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        return of(settings);
      } catch (error) {
        console.error('Error parsing saved settings:', error);
      }
    }

    // Fallback to API call (if implemented)
    const url = `${this.apiUrl}/api/settings`;
    return this.http.get<UserSettings>(url);
  }

  saveSettings(settings: UserSettings): Observable<SettingsResponse> {
    // Save to localStorage
    localStorage.setItem(this.storageKey, JSON.stringify(settings));
    
    // Also send to API (if implemented)
    const url = `${this.apiUrl}/api/settings`;
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    return this.http.post<SettingsResponse>(url, settings, { headers });
  }

  getSystemInfo(): Observable<SystemInfo> {
    const url = `${this.apiUrl}/api/system-info`;
    return this.http.get<SystemInfo>(url);
  }

  clearCache(): Observable<any> {
    const url = `${this.apiUrl}/api/clear-cache`;
    return this.http.post(url, {});
  }

  // Method to get health status
  getHealthStatus(): Observable<any> {
    const url = `${this.apiUrl}/health`;
    return this.http.get(url);
  }

  // Method to get API version
  getApiVersion(): Observable<any> {
    const url = `${this.apiUrl}/api/version`;
    return this.http.get(url);
  }

  // Method to reset settings
  resetSettings(): Observable<SettingsResponse> {
    localStorage.removeItem(this.storageKey);
    
    const url = `${this.apiUrl}/api/settings/reset`;
    return this.http.post<SettingsResponse>(url, {});
  }
}