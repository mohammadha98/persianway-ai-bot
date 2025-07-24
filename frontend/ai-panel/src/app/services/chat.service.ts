import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface ChatResponse {
  response: string;
  conversation_id?: string;
  status: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  sendMessage(message: string, conversationId?: string | null): Observable<ChatResponse> {
    const url = `${this.apiUrl}/api/chat`;
    
    const payload: ChatRequest = {
      message: message
    };
    
    if (conversationId) {
      payload.conversation_id = conversationId;
    }

    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    return this.http.post<ChatResponse>(url, payload, { headers });
  }

  // Method to get conversation history if needed
  getConversationHistory(conversationId: string): Observable<any> {
    const url = `${this.apiUrl}/api/conversations/${conversationId}`;
    return this.http.get(url);
  }

  // Method to get all conversations if needed
  getConversations(): Observable<any> {
    const url = `${this.apiUrl}/api/conversations`;
    return this.http.get(url);
  }
}