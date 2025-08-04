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

// New interfaces for the enhanced chat API
export interface QueryAnalysis {
  confidence_score: number;
  knowledge_source: string;
  requires_human_referral: boolean;
  reasoning: string;
}

export interface ResponseParameters {
  model: string;
  temperature: number;
  max_tokens: number;
  top_p: number;
}

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface EnhancedChatRequest {
  user_id: string;
  message: string;
}

export interface EnhancedChatResponse {
  query_analysis: QueryAnalysis;
  response_parameters: ResponseParameters;
  answer: string;
  conversation_history: ConversationMessage[];
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  /**
   * Send a chat message using the enhanced chat API
   * @param request - The chat request containing user_id and message
   * @returns Observable of the enhanced chat response
   */
  sendChatMessage(request: EnhancedChatRequest): Observable<EnhancedChatResponse> {
    const headers = new HttpHeaders({
      'accept': 'application/json',
      'Content-Type': 'application/json'
    });

    return this.http.post<EnhancedChatResponse>(
      `${this.apiUrl}/api/chat/`,
      request,
      { headers }
    );
  }

}