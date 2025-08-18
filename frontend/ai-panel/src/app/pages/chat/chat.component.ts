import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
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

import { ChatService, EnhancedChatResponse, QueryAnalysis, ResponseParameters, ConversationResponse, ConversationListResponse } from '../../services/chat.service';

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  queryAnalysis?: QueryAnalysis;
  responseParameters?: ResponseParameters;
  conversationHistory?: { role: string; content: string; }[];
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
    MatTooltipModule
  ],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss'
})
export class ChatComponent implements OnInit, AfterViewChecked {
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

  constructor(private chatService: ChatService) {}

  ngOnInit() {
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



  sendSuggestion(suggestion: string) {
    this.currentMessage = suggestion;
    this.sendMessage();
  }

  sendMessage() {
    if (!this.currentMessage.trim() || this.isLoading) {
      return;
    }

    // Store message content before clearing the input
    const messageContent = this.currentMessage.trim();
    
    // Add user message
    const userMessage: Message = {
      id: this.generateId(),
      content: messageContent,
      isUser: true,
      timestamp: new Date()
    };
    
    // Add to messages array and force change detection
    this.messages = [...this.messages, userMessage];
    
    // Clear input field
    this.currentMessage = '';
    this.isLoading = true;

    // Send to API using enhanced chat service with all required parameters
    const chatRequest = this.chatService.prepareChatRequest(messageContent);
    this.chatService.sendChatMessage(chatRequest)
      .subscribe({
        next: (response: EnhancedChatResponse) => {
          this.isLoading = false;
          
          // Add AI response with enhanced data
          const aiMessage: Message = {
            id: this.generateId(),
            content: response.answer || 'متأسفم، نتوانستم پاسخی تولید کنم.',
            isUser: false,
            timestamp: new Date(),
            queryAnalysis: response.query_analysis,
            responseParameters: response.response_parameters,
            conversationHistory: response.conversation_history
          };
          
          // Use immutable update pattern for better change detection
          this.messages = [...this.messages, aiMessage];
          
          // Refresh conversation history to include the new conversation
          this.loadConversationHistory();
          
          // Ensure UI updates by triggering change detection
          setTimeout(() => this.scrollToBottom(), 0);
        },
        error: (error) => {
          this.isLoading = false;
          console.error('Chat error:', error);
          
          const errorMessage: Message = {
            id: this.generateId(),
            content: 'متأسفم، خطایی در ارسال پیام رخ داد. لطفاً دوباره تلاش کنید.',
            isUser: false,
            timestamp: new Date()
          };
          
          // Use immutable update pattern for better change detection
          this.messages = [...this.messages, errorMessage];
          
          // Ensure UI updates by triggering change detection
          setTimeout(() => this.scrollToBottom(), 0);
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
    const userMessage: Message = {
      id: this.generateId(),
      content: conversation.question || '',
      isUser: true,
      timestamp: new Date(conversation.timestamp || Date.now())
    };
    
    const assistantMessage: Message = {
      id: this.generateId(),
      content: conversation.response || '',
      isUser: false,
      timestamp: new Date(conversation.timestamp || Date.now()),
      queryAnalysis: conversation.query_analysis,
      responseParameters: conversation.response_parameters
    };
    
    this.messages = [userMessage, assistantMessage];
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
                } : undefined
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
        this.messagesContainer.nativeElement.scrollTop = 
          this.messagesContainer.nativeElement.scrollHeight;
      }
    } catch (err) {
      console.error('Error scrolling to bottom:', err);
    }
  }

  private generateId(): string {
    return Math.random().toString(36).substr(2, 9);
  }

  formatTime(timestamp: Date): string {
    return timestamp.toLocaleTimeString('fa-IR', {
      hour: '2-digit',
      minute: '2-digit'
    });
  }
}