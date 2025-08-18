import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
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
  session_id?: string;
  user_email?: string;
  message: string;
}

export interface EnhancedChatResponse {
  query_analysis: QueryAnalysis;
  response_parameters: ResponseParameters;
  answer: string;
  conversation_history: ConversationMessage[];
}

// Conversation history interfaces
export interface ConversationResponse {
  id: string;
  user_id: string;
  user_email?: string;
  session_id:string;
  title?: string;
  messages: any[];
  created_at: string;
  updated_at: string;
  total_messages: number;
  is_active: boolean;
  question?: string;
  response?: string;
  timestamp?: string;
  query_analysis?: QueryAnalysis;
  response_parameters?: ResponseParameters;
}

export interface ConversationListResponse {
  conversations: ConversationResponse[];
  total_count: number;
  page: number;
  page_size: number;
  has_next: boolean;
  total?: number;
  skip?: number;
  limit?: number;
}

export interface ConversationSearchRequest {
  user_id?: string;
  start_date?: string;
  end_date?: string;
  search_text?: string;
  knowledge_source?: 'knowledge_base' | 'general_knowledge' | 'none';
  requires_human_referral?: boolean;
  min_confidence?: number;
  max_confidence?: number;
  limit?: number;
  skip?: number;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private apiUrl = environment.apiUrl || 'http://localhost:8000';
  private currentSessionId: string | null = null;

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

  /**
   * Get conversation history for a specific user
   * @param userId - The user ID to fetch conversations for
   * @param limit - Maximum number of conversations to return (default: 50)
   * @param skip - Number of conversations to skip for pagination (default: 0)
   * @returns Observable of conversation list response
   */
  getUserConversations(userId: string, limit: number = 50, skip: number = 0): Observable<ConversationListResponse> {
    const headers = new HttpHeaders({
      'accept': 'application/json'
    });

    return this.http.get<ConversationListResponse>(
      `${this.apiUrl}/api/conversations/${userId}?limit=${limit}&skip=${skip}`,
      { headers }
    );
  }

  /**
   * Get the latest conversations for a specific user
   * @param userId - The user ID to fetch conversations for
   * @param limit - Maximum number of latest conversations to return (default: 10)
   * @returns Observable of conversation responses array
   */
  getLatestUserConversations(userId: string, limit: number = 10): Observable<ConversationResponse[]> {
    const headers = new HttpHeaders({
      'accept': 'application/json'
    });

    return this.http.get<ConversationResponse[]>(
      `${this.apiUrl}/api/conversations/${userId}/latest?limit=${limit}`,
      { headers }
    );
  }

  /**
   * Generate a unique session ID
   * @returns A unique session identifier
   */
  generateSessionId(): string {
    if (!this.currentSessionId) {
      this.currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    return this.currentSessionId;
  }

  /**
   * Reset the current session (for new conversations)
   */
  resetSession(): void {
    this.currentSessionId = null;
  }

  /**
   * Get user data from localStorage
   * @returns User data object or null if not found
   */
  getUserFromStorage(): any {
    try {
      const userData = localStorage.getItem('auth_user');
      return userData ? JSON.parse(userData) : null;
    } catch (error) {
      console.error('Error parsing user data from localStorage:', error);
      return null;
    }
  }

  /**
   * Generate a consistent user ID from email address
   * @param email - User email address
   * @returns Consistent user ID based on email
   */
  generateUserIdFromEmail(email: string): string {
    // Simple hash function to create consistent user ID from email
    let hash = 0;
    for (let i = 0; i < email.length; i++) {
      const char = email.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return 'user_' + Math.abs(hash).toString(36);
  }

  /**
   * Prepare chat request with all required parameters
   * @param message - The chat message
   * @returns Enhanced chat request object
   */
  prepareChatRequest(message: string): EnhancedChatRequest {
    const user = this.getUserFromStorage();
    const userEmail = user?.email || null;
    const userId = userEmail ? this.generateUserIdFromEmail(userEmail) : 'anonymous_' + Math.random().toString(36).substr(2, 9);
    const sessionId = this.generateSessionId();

    return {
      user_id: userId,
      session_id: sessionId,
      user_email: userEmail,
      message: message
    };
  }

  /**
   * Get conversations by user email with pagination support
   * @param userEmail - The email address of the user
   * @param limit - Maximum number of conversations to return (default: 50, range: 1-100)
   * @param skip - Number of conversations to skip for pagination (default: 0)
   * @returns Observable of conversation list response
   */
  getConversationsByEmail(
    userEmail: string,
    limit: number = 50,
    skip: number = 0
  ): Observable<ConversationListResponse> {
    const headers = new HttpHeaders({
      'accept': 'application/json'
    });

    const params = new HttpParams()
      .set('limit', limit.toString())
      .set('skip', skip.toString());

    return this.http.get<ConversationListResponse>(
      `${this.apiUrl}/api/conversations/email/${encodeURIComponent(userEmail)}`,
      { headers, params }
    );
  }

  /**
   * Get conversations by session ID
   * @param sessionId - The session identifier
   * @returns Observable of conversation response array
   */
  getConversationsBySessionId(sessionId: string): Observable<ConversationResponse[]> {
    const headers = new HttpHeaders({
      'accept': 'application/json'
    });

    return this.http.get<ConversationResponse[]>(
      `${this.apiUrl}/api/conversations/session/${encodeURIComponent(sessionId)}`,
      { headers }
    );
  }

  /**
   * Search conversations based on various criteria
   * @param searchCriteria - The search criteria object
   * @returns Observable of conversation list response
   */
  searchConversations(searchCriteria: ConversationSearchRequest): Observable<ConversationListResponse> {
    const headers = new HttpHeaders({
      'accept': 'application/json',
      'Content-Type': 'application/json'
    });

    return this.http.post<ConversationListResponse>(
      `${this.apiUrl}/api/conversations/search`,
      searchCriteria,
      { headers }
    );
  }

}