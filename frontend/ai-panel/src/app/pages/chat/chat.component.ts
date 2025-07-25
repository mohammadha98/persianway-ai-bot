import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';

import { ChatService } from '../../services/chat.service';

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
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
  conversationId: string | null = null;

  constructor(private chatService: ChatService) {}

  ngOnInit() {
    this.addWelcomeMessage();
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  private addWelcomeMessage() {
    const welcomeMessage: Message = {
      id: 'welcome',
      content: 'سلام! من دستیار هوش مصنوعی کشاورزی فارسی هستم. چگونه می‌توانم به شما کمک کنم؟',
      isUser: false,
      timestamp: new Date()
    };
    // Use immutable update pattern
    this.messages = [welcomeMessage];
    
    // Ensure UI updates
    setTimeout(() => this.scrollToBottom(), 0);
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

    // Send to API
    this.chatService.sendMessage(messageContent, this.conversationId)
      .subscribe({
        next: (response) => {
          this.isLoading = false;
          
          // Add AI response
          const aiMessage: Message = {
            id: this.generateId(),
            content: response.response || 'متأسفم، نتوانستم پاسخی تولید کنم.',
            isUser: false,
            timestamp: new Date()
          };
          
          // Use immutable update pattern for better change detection
          this.messages = [...this.messages, aiMessage];
          
          // Update conversation ID if provided
          if (response.conversation_id) {
            this.conversationId = response.conversation_id;
          }
          
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
    this.conversationId = null;
    this.currentMessage = '';
    this.addWelcomeMessage();
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