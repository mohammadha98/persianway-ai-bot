import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewChecked, NgZone } from '@angular/core';
import { Subscription } from 'rxjs';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatChipsModule } from '@angular/material/chips';
import { MatBadgeModule } from '@angular/material/badge';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';
import { MatDividerModule } from '@angular/material/divider';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

import { ChatService, EnhancedChatResponse, QueryAnalysis, ResponseParameters, ConversationResponse, ConversationListResponse } from '../../services/chat.service';
import { FeedbackDialogComponent } from '../../modals/feedback-dialog/feedback-dialog.component';
import { SourcesDialogComponent } from '../../modals/sources-dialog/sources-dialog.component';
import { ContextDialogComponent } from '../../modals/context-dialog/context-dialog.component';
import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  queryAnalysis?: QueryAnalysis;
  responseParameters?: ResponseParameters;
  conversationHistory?: { role: string; content: string; }[];
  feedbackStatus?: 'approved' | 'reported' | null;
  feedbackComment?: string;
  submittingFeedback?: boolean;
  feedbackId?: string;  // Server-side feedback ID after submission
  // Server-side IDs (for feedback navigation)
  message_id?: string;
  conversation_id?: string;
  session_id?: string;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatInputModule,
    MatFormFieldModule,
    MatExpansionModule,
    MatChipsModule,
    MatBadgeModule,
    MatProgressBarModule,
    MatSidenavModule,
    MatListModule,
    MatDividerModule,
    MatTooltipModule,
    MatDialogModule,
    MatSnackBarModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss'
})
export class ChatComponent implements OnInit, AfterViewChecked, OnDestroy {
  @ViewChild('messagesContainer') messagesContainer!: ElementRef;
  @ViewChild('messageInput') messageInput!: ElementRef;

  messages: Message[] = [];
  currentMessage = '';
  isLoading = false;
  userId = '';
  showAnalysisDetails = false;

  // Conversation history properties
  conversationHistory: ConversationResponse[] = [];
  showConversationHistory = true;
  isLoadingHistory = false;
  // Active SSE stream subscription; unsubscribing aborts the underlying fetch.
  private streamSubscription: Subscription | null = null;
  // در constructor یا ngOnInit:
  isMobile = false;
  constructor(
    private chatService: ChatService,
    private dialog: MatDialog,
    private snackBar: MatSnackBar,
    private breakpointObserver: BreakpointObserver,
    private ngZone: NgZone
  ) { }

  ngOnInit() {
    this.breakpointObserver.observe([Breakpoints.Handset])
      .subscribe(result => {
        this.isMobile = result.matches;
      });
    // Initialize userId from user data
    const user = this.chatService.getUserFromStorage();
    if (user?.email) {
      this.userId = this.chatService.generateUserIdFromEmail(user.email);
    } else {
      this.userId = 'anonymous_' + Math.random().toString(36).substr(2, 9);
    }
    this.loadConversationHistory();
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  ngOnDestroy() {
    // Abort any in-flight SSE stream when the component is destroyed
    // (e.g. user navigates away). The service teardown aborts the fetch.
    if (this.streamSubscription) {
      this.streamSubscription.unsubscribe();
      this.streamSubscription = null;
    }
  }



  sendSuggestion(suggestion: string) {
    this.currentMessage = suggestion;
    this.sendMessage();
  }

  sendMessage() {
    if (!this.currentMessage.trim() || this.isLoading) {
      console.log('[CHAT] sendMessage blocked - loading:', this.isLoading, 'empty:', !this.currentMessage.trim());
      return;
    }

    console.log('[CHAT] sendMessage called with:', this.currentMessage);

    const messageContent = this.currentMessage.trim();

    const userMessage: Message = {
      id: this.generateId(),
      content: messageContent,
      isUser: true,
      timestamp: new Date()
    };

    this.messages = [...this.messages, userMessage];
    console.log('[CHAT] Added user message, total messages:', this.messages.length);

    this.currentMessage = '';
    this.isLoading = true;
    console.log('[CHAT] Set isLoading to true');

    const chatRequest = this.chatService.prepareChatRequest(messageContent);
    console.log('[CHAT] Prepared chat request:', chatRequest);

    let fullAnswer = '';
    let metadata: any = null;
    let aiMessageId: string | null = null;
    let typedText = '';
    let typingTimer: any = null;

    const ensureAiMessage = (): Message | undefined => {
      if (!aiMessageId) {
        aiMessageId = this.generateId();
        const aiMessage: Message = {
          id: aiMessageId,
          content: '',
          isUser: false,
          timestamp: new Date(),
          queryAnalysis: undefined,
          responseParameters: undefined,
          conversationHistory: undefined,
          message_id: undefined,
          conversation_id: undefined,
          session_id: undefined
        };
        this.messages = [...this.messages, aiMessage];
        console.log('[CHAT] Added AI message placeholder with id:', aiMessageId);
      }

      return this.messages.find(m => m.id === aiMessageId);
    };

    const flushTypingEffect = () => {
      if (typingTimer) {
        clearTimeout(typingTimer);
      }

      const targetMessage = ensureAiMessage();
      if (!targetMessage) {
        return;
      }

      let visibleLength = targetMessage.content.length;
      const targetText = typedText;

      const tick = () => {
        const nextVisibleLength = Math.min(visibleLength + 18, targetText.length);
        targetMessage.content = targetText.slice(0, nextVisibleLength);
        this.messages = [...this.messages];
        visibleLength = nextVisibleLength;
        this.scrollToBottom();

        if (nextVisibleLength < targetText.length) {
          typingTimer = setTimeout(() => this.ngZone.run(tick), 20);
        } else {
          typingTimer = null;
          if (targetMessage) {
            targetMessage.content = targetText;
            this.messages = [...this.messages];
          }
        }
      };

      tick();
    };

    console.log('[CHAT] Subscribing to streamChatMessage');
    // Dispose any previous stream subscription (aborts its fetch) before
    // starting a new one.
    if (this.streamSubscription) {
      this.streamSubscription.unsubscribe();
      this.streamSubscription = null;
    }
    this.streamSubscription = this.chatService.streamChatMessage(chatRequest)
      .subscribe({
        next: (event) => {
          console.log('[CHAT] Received event from stream:', event);

          if (event.type === 'metadata') {
            console.log('[CHAT] Processing metadata event');
            metadata = event.data;
            console.log('[CHAT] Metadata:', metadata);

            const messageToUpdate = ensureAiMessage();
            console.log('[CHAT] Found message to update:', !!messageToUpdate);

            if (messageToUpdate) {
              messageToUpdate.queryAnalysis = metadata.query_analysis;
              messageToUpdate.responseParameters = metadata.response_parameters;
              messageToUpdate.conversationHistory = metadata.conversation_history;
              console.log('[CHAT] Updated message with metadata');
            }
          } else if (event.type === 'chunk') {
            console.log('[CHAT] Processing chunk event, content length:', event.content?.length);
            fullAnswer += event.content;
            typedText = fullAnswer;
            console.log('[CHAT] Full answer length so far:', fullAnswer.length);

            this.ngZone.run(() => {
              flushTypingEffect();
            });
          } else if (event.type === 'done') {
            console.log('[CHAT] Processing done event');
            this.isLoading = false;
            const messageToUpdate = ensureAiMessage();
            if (messageToUpdate) {
              messageToUpdate.content = fullAnswer || messageToUpdate.content || '';
              messageToUpdate.message_id = event.message_id;
              messageToUpdate.conversation_id = event.conversation_id;
              messageToUpdate.session_id = event.session_id || chatRequest.session_id;
              messageToUpdate.conversationHistory = event.conversation_history;
              this.messages = [...this.messages];
              console.log('[CHAT] Updated message with server-side IDs');
            }
            console.log('[CHAT] Loading conversation history');
            this.loadConversationHistory();
          } else {
            console.warn('[CHAT] Unknown event type:', event.type);
          }
        },
        error: (error) => {
          console.error('[CHAT] Stream error:', error);
          this.isLoading = false;

          // Stop the typing effect immediately so it cannot keep writing
          // into the message after the stream failed.
          if (typingTimer) {
            clearTimeout(typingTimer);
            typingTimer = null;
          }

          const errorMessageText = (error && error.message)
            ? error.message
            : 'متأسفم، خطایی در ارسال پیام رخ داد. لطفاً دوباره تلاش کنید.';

          if (fullAnswer.length > 0) {
            // Keep the partial answer visible and report the error via the
            // snackbar, so error text is never glued into the answer body.
            const partialMessage = ensureAiMessage();
            if (partialMessage) {
              partialMessage.content = fullAnswer;
              this.messages = [...this.messages];
              console.log('[CHAT] Kept partial answer after error, length:', fullAnswer.length);
            }
          } else if (error && error.code === 'stream_incomplete') {
            // Stream ended without a `done` event and nothing was received.
            const messageToUpdate = ensureAiMessage();
            if (messageToUpdate) {
              messageToUpdate.content = errorMessageText;
              this.messages = [...this.messages];
            }
          } else {
            // No content received at all: replace the placeholder with an
            // explicit error message.
            if (aiMessageId) {
              this.messages = this.messages.filter(m => m.id !== aiMessageId);
            }

            const errorMessage: Message = {
              id: this.generateId(),
              content: errorMessageText,
              isUser: false,
              timestamp: new Date()
            };

            this.messages = [...this.messages, errorMessage];
            console.log('[CHAT] Added error message');
          }

          this.snackBar.open(errorMessageText, 'بستن', {
            duration: 4000,
            horizontalPosition: 'center',
            verticalPosition: 'bottom',
            panelClass: ['error-snackbar']
          });
        },
        complete: () => {
          console.log('[CHAT] Stream subscription completed');
          this.isLoading = false;
          this.streamSubscription = null;
        }
      });
  }

  onKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  clearChat() {
    // Abort any in-flight stream before clearing the chat.
    if (this.streamSubscription) {
      this.streamSubscription.unsubscribe();
      this.streamSubscription = null;
    }
    this.isLoading = false;
    this.messages = [];
    this.currentMessage = '';

    // Reset session for new conversation
    this.chatService.resetSession();

    // Reinitialize userId from user data
    const user = this.chatService.getUserFromStorage();
    if (user?.email) {
      this.userId = this.chatService.generateUserIdFromEmail(user.email);
    } else {
      this.userId = 'anonymous_' + Math.random().toString(36).substr(2, 9);
    }
  }

  toggleAnalysisDetails() {
    this.showAnalysisDetails = !this.showAnalysisDetails;
  }

  // Conversation history is now permanently visible
  // No toggle functionality needed

  loadConversationHistory() {
    this.isLoadingHistory = true;
    const user = this.chatService.getUserFromStorage();
    debugger
    if (user?.email) {
      // Use the new API endpoint to get conversations by email
      this.chatService.getConversationsByEmail(user.email, 20, 0)
        .subscribe({
          next: (response) => {
            this.conversationHistory = response.conversations;
            console.log(...this.conversationHistory)
            this.isLoadingHistory = false;
          },
          error: (error) => {
            console.error('Failed to load conversation history:', error);
            this.isLoadingHistory = false;
          }
        });
    } else {
      // Fallback: no email available, clear history
      this.conversationHistory = [];
      this.isLoadingHistory = false;
    }
  }

  loadConversation(conversation: ConversationResponse) {
    // Clear current messages
    this.messages = [];

    // Add the selected conversation to messages
    conversation.messages.forEach(msg => {
      const message: Message = {
        id: this.generateId(),
        content: msg.content || '',
        isUser: msg.role === 'user',
        timestamp: new Date(msg.timestamp || Date.now()),
        queryAnalysis: msg.role === 'assistant' ? {
          confidence_score: msg.confidence_score || 0,
          knowledge_source: msg.knowledge_source || 'unknown',
          requires_human_referral: msg.requires_human_referral || false,
          reasoning: ''
        } : undefined,
        responseParameters: msg.role === 'assistant' ? {
          model: 'default',
          temperature: 0.7,
          max_tokens: 1000,
          top_p: 1
        } : undefined,
        // Store server-side IDs for feedback navigation
        message_id: (msg as any).message_id,
        conversation_id: conversation.conversation_id || conversation.id,
        session_id: conversation.session_id
      };

      this.messages.push(message);
    });
    this.showConversationHistory = false;
  }

  loadConversationsBySessionId(sessionId: string) {
    this.isLoadingHistory = true;
    this.chatService.getConversationsBySessionId(sessionId)
      .subscribe({
        next: (conversations) => {
          // Load all conversations from this session into messages
          this.messages = [];
          conversations.forEach(conversation => {
            // Get the conversation_id from the document for feedback navigation
            const conversationId = (conversation as any).conversation_id;

            // Process each message in the conversation
            conversation.messages?.forEach(msg => {
              const message: Message = {
                id: this.generateId(),
                content: msg.content || '',
                isUser: msg.role === 'user',
                timestamp: new Date(msg.timestamp || Date.now()),
                queryAnalysis: msg.role === 'assistant' ? {
                  confidence_score: msg.confidence_score || 0,
                  knowledge_source: msg.knowledge_source || 'unknown',
                  requires_human_referral: msg.requires_human_referral || false,
                  reasoning: ''
                } : undefined,
                responseParameters: msg.role === 'assistant' ? {
                  model: 'default',
                  temperature: 0.7,
                  max_tokens: 1000,
                  top_p: 1
                } : undefined,
                // Store server-side IDs for feedback navigation
                message_id: (msg as any).message_id,
                conversation_id: conversationId,
                session_id: sessionId
              };

              this.messages.push(message);
            });
          });
          this.isLoadingHistory = false;
          this.showConversationHistory = false;
        },
        error: (error) => {
          console.error('Failed to load conversations by session ID:', error);
          this.isLoadingHistory = false;
        }
      });
  }

  formatConversationDate(timestamp: string): string {
    const date = new Date(timestamp);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) {
      return 'امروز';
    } else if (diffDays === 2) {
      return 'دیروز';
    } else if (diffDays <= 7) {
      return `${diffDays} روز پیش`;
    } else {
      return date.toLocaleDateString('fa-IR');
    }
  }

  truncateText(text: string, maxLength: number = 50): string {
    if (text.length <= maxLength) {
      return text;
    }
    return text.substring(0, maxLength) + '...';
  }

  getConfidenceColor(score: number): string {
    if (score >= 0.8) return '#4CAF50';
    if (score >= 0.6) return '#FF9800';
    return '#F44336';
  }

  getConfidenceText(score: number): string {
    if (score >= 0.8) return 'بالا';
    if (score >= 0.6) return 'متوسط';
    return 'پایین';
  }

  getKnowledgeSourceText(source: string): string {
    const sources: { [key: string]: string } = {
      'general_knowledge': 'دانش عمومی',
      'agriculture_knowledge': 'دانش کشاورزی',
      'specialized_knowledge': 'دانش تخصصی',
      'external_source': 'منبع خارجی'
    };
    return sources[source] || source;
  }

  private scrollToBottom() {
    try {
      if (this.messagesContainer) {
        const scrollHeight = this.messagesContainer.nativeElement.scrollHeight;
        const currentScroll = this.messagesContainer.nativeElement.scrollTop;
        console.log('[CHAT] scrollToBottom - scrollHeight:', scrollHeight, 'currentScroll:', currentScroll);
        
        this.messagesContainer.nativeElement.scrollTop = scrollHeight;
        console.log('[CHAT] scrollToBottom complete - new scroll:', this.messagesContainer.nativeElement.scrollTop);
      } else {
        console.warn('[CHAT] messagesContainer not available for scrolling');
      }
    } catch (err) {
      console.error('[CHAT] Error scrolling to bottom:', err);
    }
  }

  private generateId(): string {
    return Math.random().toString(36).substr(2, 9);
  }

  /**
   * Log diagnostic information for debugging streaming issues
   * Can be called from browser console with: ng.probe(document.querySelector('app-chat')).componentInstance.diagnose()
   */
  diagnose(): void {
    console.log('=== CHAT COMPONENT DIAGNOSTICS ===');
    console.log('Messages count:', this.messages.length);
    console.log('Messages:', this.messages);
    console.log('isLoading:', this.isLoading);
    console.log('userId:', this.userId);
    console.log('currentMessage:', this.currentMessage);
    console.log('messagesContainer available:', !!this.messagesContainer);
    if (this.messagesContainer) {
      console.log('messagesContainer scrollHeight:', this.messagesContainer.nativeElement.scrollHeight);
      console.log('messagesContainer scrollTop:', this.messagesContainer.nativeElement.scrollTop);
      console.log('messagesContainer clientHeight:', this.messagesContainer.nativeElement.clientHeight);
    }
    console.log('showAnalysisDetails:', this.showAnalysisDetails);
    console.log('conversationHistory count:', this.conversationHistory.length);
    console.log('=== END DIAGNOSTICS ===');
  }

  formatTime(timestamp: Date): string {
    return timestamp.toLocaleTimeString('fa-IR', {
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  /**
   * Open a dialog to show the sources used by the AI
   * @param message - The AI message to inspect
   */
  viewSources(message: Message): void {
    // Use server-side message_id (fallback to client ID if not available)
    const messageId = message.message_id || message.id;

    if (!message.message_id) {
      this.snackBar.open('شناسه سروری این پیام در دسترس نیست. لطفاً مکالمه را دوباره بارگذاری کنید.', 'بستن', {
        duration: 4000,
        horizontalPosition: 'center',
        verticalPosition: 'bottom',
        panelClass: ['error-snackbar']
      });
      return;
    }

    // Open dialog with loading state
    const dialogRef = this.dialog.open(SourcesDialogComponent, {
      width: '700px',
      maxWidth: '95vw',
      data: {
        sources: [],
        isLoading: true
      },
      panelClass: 'sources-dialog'
    });

    this.chatService.getMessageSources(messageId).subscribe({
      next: (response) => {
        // Update dialog data with the sources
        dialogRef.componentInstance.data.sources = response.sources || [];
        dialogRef.componentInstance.data.isLoading = false;
      },
      error: (error) => {
        console.error('Error fetching sources:', error);
        dialogRef.componentInstance.data.isLoading = false;
        this.snackBar.open('خطا در دریافت منابع. لطفاً دوباره تلاش کنید.', 'بستن', {
          duration: 4000,
          horizontalPosition: 'center',
          verticalPosition: 'bottom',
          panelClass: ['error-snackbar']
        });
      }
    });
  }

  /**
   * Open a dialog to show the context and prompt sent to the LLM
   * @param message - The AI message to inspect
   */
  viewContext(message: Message): void {
    // Use server-side message_id (fallback to client ID if not available)
    const messageId = message.message_id || message.id;

    if (!message.message_id) {
      this.snackBar.open('شناسه سروری این پیام در دسترس نیست. لطفاً مکالمه را دوباره بارگذاری کنید.', 'بستن', {
        duration: 4000,
        horizontalPosition: 'center',
        verticalPosition: 'bottom',
        panelClass: ['error-snackbar']
      });
      return;
    }

    // Open dialog with loading state
    const dialogRef = this.dialog.open(ContextDialogComponent, {
      width: '800px',
      maxWidth: '95vw',
      data: {
        data: null,
        isLoading: true
      },
      panelClass: 'context-dialog'
    });

    this.chatService.getMessageContext(messageId).subscribe({
      next: (response) => {
        // Update dialog data with the context info
        dialogRef.componentInstance.data.data = response.data || null;
        dialogRef.componentInstance.data.isLoading = false;
      },
      error: (error) => {
        console.error('Error fetching context:', error);
        dialogRef.componentInstance.data.isLoading = false;
        const message = error.status === 404
          ? 'context برای این پیام موجود نیست'
          : 'خطا در دریافت Context. لطفاً دوباره تلاش کنید.';
        this.snackBar.open(message, 'بستن', {
          duration: 4000,
          horizontalPosition: 'center',
          verticalPosition: 'bottom',
          panelClass: ['error-snackbar']
        });
      }
    });
  }

  /**
   * Approve an AI message - mark it as correct feedback
   * @param message - The AI message to approve
   */
  approveMessage(message: Message): void {
    if (message.feedbackStatus === 'approved' || message.submittingFeedback) {
      return;
    }

    // Use server-side message_id (fallback to client ID if not available)
    const messageId = message.message_id || message.id;
    const conversationId = message.conversation_id;

    // Find the user question that preceded this message
    const messageIndex = this.messages.findIndex(m => m.id === message.id);
    let userQuestion = '';
    for (let i = messageIndex - 1; i >= 0; i--) {
      if (this.messages[i].isUser) {
        userQuestion = this.messages[i].content;
        break;
      }
    }

    message.submittingFeedback = true;
    this.chatService.submitFeedback(
      messageId,
      'approve',
      '',
      message.content,
      userQuestion,
      conversationId
    ).subscribe({
      next: (response) => {
        message.feedbackStatus = 'approved';
        message.feedbackId = response?.feedback_id;
        message.submittingFeedback = false;
        this.snackBar.open('بازخورد شما با موفقیت ثبت شد ✓', 'بستن', {
          duration: 3000,
          horizontalPosition: 'center',
          verticalPosition: 'bottom',
          panelClass: ['success-snackbar']
        });
      },
      error: (error) => {
        console.error('Error submitting feedback:', error);
        message.submittingFeedback = false;
        this.snackBar.open('خطا در ثبت بازخورد. لطفاً دوباره تلاش کنید.', 'بستن', {
          duration: 4000,
          horizontalPosition: 'center',
          verticalPosition: 'bottom',
          panelClass: ['error-snackbar']
        });
      }
    });
  }

  /**
   * Open a dialog to report an issue with an AI message
   * @param message - The AI message to report
   */
  reportMessage(message: Message): void {
    if (message.feedbackStatus === 'reported' || message.submittingFeedback) {
      return;
    }

    // Use server-side message_id (fallback to client ID if not available)
    const messageId = message.message_id || message.id;
    const conversationId = message.conversation_id;

    // Find the user question that preceded this message
    const messageIndex = this.messages.findIndex(m => m.id === message.id);
    let userQuestion = '';
    for (let i = messageIndex - 1; i >= 0; i--) {
      if (this.messages[i].isUser) {
        userQuestion = this.messages[i].content;
        break;
      }
    }

    const dialogRef = this.dialog.open(FeedbackDialogComponent, {
      width: '500px',
      maxWidth: '95vw',
      data: {
        messageContent: message.content,
        question: userQuestion,
        messageId: messageId,
        conversationId: conversationId,
        mode: 'report'
      },
      panelClass: 'feedback-dialog'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result && result.comment) {
        message.submittingFeedback = true;
        this.chatService.submitFeedback(
          messageId,
          'report',
          result.comment,
          message.content,
          userQuestion,
          conversationId,
          result.category
        ).subscribe({
          next: (response) => {
            message.feedbackStatus = 'reported';
            message.feedbackId = response?.feedback_id;
            message.feedbackComment = result.comment;
            message.submittingFeedback = false;
            this.snackBar.open('گزارش شما با موفقیت ثبت شد. متشکریم!', 'بستن', {
              duration: 3000,
              horizontalPosition: 'center',
              verticalPosition: 'bottom',
              panelClass: ['success-snackbar']
            });
          },
          error: (error) => {
            console.error('Error submitting report:', error);
            message.submittingFeedback = false;
            this.snackBar.open('خطا در ثبت گزارش. لطفاً دوباره تلاش کنید.', 'بستن', {
              duration: 4000,
              horizontalPosition: 'center',
              verticalPosition: 'bottom',
              panelClass: ['error-snackbar']
            });
          }
        });
      }
    });
  }
}