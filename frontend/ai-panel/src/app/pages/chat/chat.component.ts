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

import { ChatService, EnhancedChatResponse, QueryAnalysis, ResponseParameters } from '../../services/chat.service';

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
    MatProgressBarModule
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
  userId = 'user_' + Math.random().toString(36).substr(2, 9);
  showAnalysisDetails = false;

  constructor(private chatService: ChatService) {}

  ngOnInit() {
  
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

    // Send to API using enhanced chat service
    this.chatService.sendChatMessage({ user_id: this.userId, message: messageContent })
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
    this.userId = 'user_' + Math.random().toString(36).substr(2, 9);
 
  }

  toggleAnalysisDetails() {
    this.showAnalysisDetails = !this.showAnalysisDetails;
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