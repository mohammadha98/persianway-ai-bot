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
  message_id?: string;        // Server-side ID for this AI message (for feedback)
  conversation_id?: string;   // Server-side ID for the conversation (for navigation)
  session_id?: string;        // Session ID for the conversation
  feedback?: any;             // Existing feedback if any
}

// Conversation history interfaces
export interface ConversationResponse {
  id: string;
  conversation_id?: string;
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
   * Internal helper: consume an SSE endpoint via fetch + ReadableStream.
   *
   * - Buffers partial SSE frames split across network chunks until a full
   *   event (delimited by a blank line, "\n\n") is received.
   * - Emits every parsed JSON event (metadata / chunk / done / error).
   * - Completes on `done`; errors with {message, code} on `error`,
   *   on stream end without `done` (code: stream_incomplete), or on
   *   network failures.
   * - Aborts the request on unsubscribe (navigation away / new message).
   */
  private sseFetchStream(url: string): Observable<any> {
    console.log('[STREAM] Starting SSE fetch stream:', url);

    return new Observable(observer => {
      const controller = new AbortController();
      let terminated = false; // true once done/error has been emitted

      fetch(url, {
        method: 'GET',
        headers: { 'accept': 'text/event-stream' },
        signal: controller.signal
      })
        .then(async response => {
          if (!response.ok || !response.body) {
            terminated = true;
            observer.error({
              message: 'خطا در برقراری ارتباط با سرور. لطفاً دوباره تلاش کنید.',
              code: 'http_' + response.status
            });
            return;
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder('utf-8');
          let buffer = '';

          const processFrame = (frame: string) => {
            // An SSE frame may contain one or more "data:" lines.
            const dataLines = frame
              .split('\n')
              .filter(line => line.startsWith('data:'))
              .map(line => line.slice(5).trimStart());
            if (dataLines.length === 0) {
              return;
            }
            const payload = dataLines.join('\n');
            if (!payload) {
              return;
            }

            // Explicit end-of-stream marker written by the API after the JSON
            // `done` event. It is deliberately not JSON, so it is recognised
            // before parsing: `done` has already completed the observable, and
            // this only confirms that the server closed the stream on purpose
            // instead of dropping the connection mid-answer.
            if (payload === '[DONE]') {
              terminated = true;
              return;
            }

            let event: any;
            try {
              event = JSON.parse(payload);
            } catch (parseError) {
              console.error('[STREAM] Failed to parse SSE payload:', payload, parseError);
              return; // skip malformed frame, do not kill the stream
            }

            observer.next(event);

            if (event.type === 'done') {
              terminated = true;
              observer.complete();
            } else if (event.type === 'error') {
              terminated = true;
              observer.error({
                message: event.message || 'خطای غیرمنتظره رخ داد. لطفاً دوباره تلاش کنید.',
                code: event.code || 'stream_internal_error'
              });
            }
          };

          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) {
                break;
              }
              buffer += decoder.decode(value, { stream: true });

              // SSE events are separated by blank lines; process every
              // complete frame and keep any partial remainder buffered.
              let sepIndex = buffer.indexOf('\n\n');
              while (sepIndex !== -1) {
                const frame = buffer.slice(0, sepIndex);
                buffer = buffer.slice(sepIndex + 2);
                processFrame(frame);
                if (terminated) {
                  try { await reader.cancel(); } catch (e) { /* ignore */ }
                  return;
                }
                sepIndex = buffer.indexOf('\n\n');
              }
            }

            // Server closed the connection.
            if (!terminated) {
              observer.error({
                message: 'اتصال پیش از دریافت پاسخ کامل قطع شد.',
                code: 'stream_incomplete'
              });
            } else {
              observer.complete();
            }
          } catch (readError) {
            if (!terminated) {
              observer.error({
                message: 'خطا در دریافت پاسخ. لطفاً دوباره تلاش کنید.',
                code: 'stream_read_error'
              });
            }
          }
        })
        .catch(err => {
          if (err && err.name === 'AbortError') {
            // Aborted by teardown (unsubscribe/navigation) — finish silently.
            observer.complete();
            return;
          }
          if (!terminated) {
            observer.error({
              message: 'خطا در اتصال به سرور. لطفاً اتصال اینترنت خود را بررسی کنید.',
              code: 'network_error'
            });
          }
        });

      // Teardown: abort the fetch when the subscription is disposed.
      return () => {
        controller.abort();
      };
    });
  }

  /**
   * Stream a chat message response using Server-Sent Events (SSE)
   * @param request - The chat request containing user_id and message
   * @returns Observable of streaming events
   */
  streamChatMessage(request: EnhancedChatRequest): Observable<any> {
    console.log('[STREAM] Starting streamChatMessage with request:', request);

    const url = `${this.apiUrl}/api/chat/stream?` + new URLSearchParams({
      user_id: request.user_id,
      session_id: request.session_id || '',
      message: request.message,
      user_email: request.user_email || ''
    }).toString();

    return this.sseFetchStream(url);
  }

  /**
   * Stream a chat message response with spell correction using Server-Sent Events (SSE)
   * @param request - The chat request containing user_id and message
   * @returns Observable of streaming events
   */
  streamChatMessageWithCorrection(request: EnhancedChatRequest): Observable<any> {
    console.log('[STREAM] Starting streamChatMessageWithCorrection with request:', request);

    const url = `${this.apiUrl}/api/chat/stream/corrected?` + new URLSearchParams({
      user_id: request.user_id,
      session_id: request.session_id || '',
      message: request.message,
      user_email: request.user_email || ''
    }).toString();

    return this.sseFetchStream(url);
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
    console.log('[SERVICE] prepareChatRequest called with message:', message);
    const user = this.getUserFromStorage();
    console.log('[SERVICE] Current user:', user);
    
    const userEmail = user?.email || null;
    const userId = userEmail ? this.generateUserIdFromEmail(userEmail) : 'anonymous_' + Math.random().toString(36).substr(2, 9);
    const sessionId = this.generateSessionId();
    
    console.log('[SERVICE] Generated userId:', userId);
    console.log('[SERVICE] Generated sessionId:', sessionId);

    const request: EnhancedChatRequest = {
      user_id: userId,
      session_id: sessionId,
      user_email: userEmail,
      message: message
    };
    
    console.log('[SERVICE] Prepared chat request:', request);
    return request;
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

  /**
   * Submit feedback (approve, report, edit, or add_to_kb) for an AI message.
   * Integrates with the new backend feedback API.
   *
   * @param messageId - The unique identifier of the AI message
   * @param feedback - The feedback type: 'approve', 'report', 'edit', or 'add_to_kb'
   * @param comment - Optional comment (required for 'report' and 'edit')
   * @param messageContent - The content of the AI message
   * @param question - The user's question
   * @param conversationId - Optional conversation ID for navigation
   * @param category - Optional issue category (e.g., 'incorrect_info', 'incomplete')
   * @param editedAnswer - Optional edited answer (for 'edit' feedback)
   * @returns Observable of the feedback response
   */
  submitFeedback(
    messageId: string,
    feedback: 'approve' | 'report' | 'edit' | 'add_to_kb',
    comment: string,
    messageContent: string,
    question: string,
    conversationId?: string,
    category?: string,
    editedAnswer?: string
  ): Observable<any> {
    const headers = new HttpHeaders({
      'accept': 'application/json',
      'Content-Type': 'application/json'
    });

    const body = {
      message_id: messageId,
      conversation_id: conversationId || null,
      feedback: feedback,
      comment: comment || '',
      category: category || null,
      user_email: this.getUserFromStorage()?.email || null,
      user_id: this.generateUserIdFromEmail(this.getUserFromStorage()?.email || 'anonymous'),
      message_content: messageContent,
      question: question,
      edited_answer: editedAnswer || null,
      timestamp: new Date().toISOString()
    };

    return this.http.post<any>(
      `${this.apiUrl}/api/chat/feedback`,
      body,
      { headers }
    );
  }

  /**
   * Get the sources that the AI used to generate a response
   * @param messageId - The unique identifier of the message
   * @returns Observable of the sources list
   */
  getMessageSources(messageId: string): Observable<any> {
    const headers = new HttpHeaders({
      'accept': 'application/json'
    });

    return this.http.get<any>(
      `${this.apiUrl}/api/conversations/messages/${encodeURIComponent(messageId)}`,
      { headers }
    );
  }

  /**
   * Get the prompt and context that was sent to the LLM
   * @param messageId - The unique identifier of the message
   * @returns Observable of the prompt data
   */
  getMessageContext(messageId: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/api/chat/context/${messageId}`);
    /* return new Observable(observer => {
      setTimeout(() => {
        observer.next({
          success: true,
          data: {
            system_prompt: `شما یک دستیار هوشمند تخصصی در حوزه کشاورزی هستید. 
وظیفه شما پاسخگویی به سوالات کاربران با استفاده از اطلاعات موجود در پایگاه دانش است.
همیشه پاسخ‌های دقیق، مختصر و مفید ارائه دهید.
اگر اطلاعات کافی ندارید، صادقانه بگویید و کاربر را به کارشناسان ارجاع دهید.

زبان پاسخ‌گویی: فارسی
لحن: رسمی و محترمانه
حداکثر طول پاسخ: ۵۰۰ کلمه`,

            user_query: 'نحوه کاشت گوجه فرنگی چیست؟',

            retrieved_context: `[سند ۱] راهنمای کشت گوجه فرنگی (صفحه ۱۲، امتیاز: ۰.۹۲):
گوجه فرنگی گیاهی گرمادوست است که به خاک‌های غنی از مواد آلی نیاز دارد. بهترین زمان کاشت آن در فصل بهار است. دمای مناسب برای رشد ۲۰ تا ۲۵ درجه سانتی‌گراد است. آبیاری منظم و کافی ضروری است.

[سند ۲] اصول باغبانی مدرن (صفحه ۸، امتیاز: ۰.۸۵):
برای افزایش بهره‌وری، استفاده از سیستم‌های آبیاری قطره‌ای توصیه می‌شود. این روش تا ۴۰٪ در مصرف آب صرفه‌جویی می‌کند. فاصله کاشت نهال‌ها باید ۵۰ سانتی‌متر باشد.

[سند ۳] پرسش و پاسخ - فصل کاشت (امتیاز: ۰.۷۸):
س: بهترین فصل کاشت گوجه فرنگی چه زمانی است؟
ج: فصل بهار پس از پایان سرمای زمستان، معمولاً از اواسط فروردین تا اوایل اردیبهشت.`,

            conversation_history: [
              { role: 'user', content: 'سلام' },
              { role: 'assistant', content: 'سلام! چطور می‌توانم کمکتان کنم؟' },
              { role: 'user', content: 'می‌خواهم در مورد کشاورزی سوال بپرسم' },
              { role: 'assistant', content: 'بله، بفرمایید سوالتان را بپرسید.' }
            ],

            model_parameters: {
              model: 'qwen/qwen3-32b',
              temperature: 0.1,
              max_tokens: 800,
              top_p: 1.0
            },

            full_prompt: `=== SYSTEM PROMPT ===
شما یک دستیار هوشمند تخصصی در حوزه کشاورزی هستید...
[متن کامل سیستم پرامپت]

=== CONTEXT (RETRIEVED DOCUMENTS) ===
[سند ۱] راهنمای کشت گوجه فرنگی...
[سند ۲] اصول باغبانی مدرن...
[سند ۳] پرسش و پاسخ - فصل کاشت...

=== CONVERSATION HISTORY ===
user: سلام
assistant: سلام! چطور می‌توانم کمکتان کنم؟
user: می‌خواهم در مورد کشاورزی سوال بپرسم
assistant: بله، بفرمایید سوالتان را بپرسید.

=== USER QUESTION ===
نحوه کاشت گوجه فرنگی چیست؟

=== INSTRUCTIONS ===
بر اساس اطلاعات ارائه شده در Context، به سوال کاربر پاسخ دهید.
اگر اطلاعات کافی نیست، صادقانه بگویید.`,

            timestamp: new Date().toISOString()
          }
        });
        observer.complete();
      }, 500);
    }); */
  }

}