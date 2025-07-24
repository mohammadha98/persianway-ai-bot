import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
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
    MatProgressSpinnerModule
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
    this.messages.push(welcomeMessage);
  }

  sendMessage() {
    if (!this.currentMessage.trim() || this.isLoading) {
      return;
    }

    // Add user message
    const userMessage: Message = {
      id: this.generateId(),
      content: this.currentMessage,
      isUser: true,
      timestamp: new Date()
    };
    this.messages.push(userMessage);

    const messageToSend = this.currentMessage;
    this.currentMessage = '';
    this.isLoading = true;

    // Send to API
    this.chatService.sendMessage(messageToSend, this.conversationId)
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
          this.messages.push(aiMessage);
          
          // Update conversation ID if provided
          if (response.conversation_id) {
            this.conversationId = response.conversation_id;
          }
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
          this.messages.push(errorMessage);
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