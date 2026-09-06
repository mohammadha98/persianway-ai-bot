import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { ConversationResponse } from '../../services/chat.service';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatDialogModule } from '@angular/material/dialog';
import { ContributeService } from '../../services/contribute.service';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatTooltipModule } from '@angular/material/tooltip';

@Component({
  selector: 'app-conversation-details-modal',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatIconModule,
    MatButtonModule,
    MatDialogModule,
    MatSnackBarModule,
    MatTooltipModule
  ],
  templateUrl: './conversation-details-modal.component.html',
  styleUrl: './conversation-details-modal.component.scss'
})
export class ConversationDetailsModalComponent {
  conversation: ConversationResponse;

  constructor(
    public dialogRef: MatDialogRef<ConversationDetailsModalComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { conversation: ConversationResponse },
    private contributeService: ContributeService,
    private snackBar: MatSnackBar
  ) {
    this.conversation = data.conversation;
  }

  onClose(): void {
    this.dialogRef.close();
  }



  addToKnowledgeBase(message: any, index: number) {
    if (index > 0) {
      const previousMessage = this.conversation.messages[index - 1];
      if (previousMessage.role === 'user') {
        const formData = new FormData();
        formData.append('title', previousMessage.content);
        formData.append('content', message.content);
        formData.append('source', `Conversation: ${this.conversation.session_id}`);
        formData.append('meta_tags', 'conversation, qa');
        formData.append('author_name', this.conversation.user_email || 'Unknown');
        formData.append('additional_references', '');

        this.contributeService.submitContribution(formData).subscribe({
          next: (response) => {
            this.snackBar.open('پرسش و پاسخ با موفقیت به دانش پایگاه اضافه شد.', 'بستن', {
              duration: 5000,
              panelClass: ['success-snackbar']
            });
          },
          error: (error) => {
            console.error('Error adding to knowledge base:', error);
            this.snackBar.open('خطا در افزودن به پایگاه دانش. لطفاً دوباره تلاش کنید.', 'بستن', {
              duration: 5000,
              panelClass: ['error-snackbar']
            });
          }
        });
      } else {
        this.snackBar.open('پیام قبلی یک سوال از کاربر نیست.', 'بستن', {
          duration: 3000,
          panelClass: ['warning-snackbar']
        });
      }
    } else {
      this.snackBar.open('هیچ پیام قبلی برای تشکیل یک جفت پرسش و پاسخ وجود ندارد.', 'بستن', {
        duration: 3000,
        panelClass: ['warning-snackbar']
      });
    }
  }
//   {
//     "role": "assistant",
//     "content": "تشخیص لبه در پردازش تصاویر دیجیتال به فرآیندی اطلاق می‌شود که در آن لبه‌ها یا تغییرات ناگهانی در شدت روشنایی تصویر شناسایی می‌شوند. این فرآیند یکی از مراحل اساسی در تحلیل و پردازش تصاویر است و به منظور استخراج ویژگی‌ها و اطلاعات مهم از تصاویر به کار می‌رود. روش‌های مختلفی برای تشخیص لبه وجود دارد که شامل فیلترهای مختلف و الگوریتم‌های متنوعی مانند فیلتر سوبل، فیلتر پریویت، فیلتر رابرتز و الگوریتم کنی می‌شود. هر یک از این روش‌ها دارای مزایا و معایب خاص خود هستند و انتخاب روش مناسب به نوع تصویر و کاربرد خاص بستگی دارد.",
//     "timestamp": "2025-08-17T15:19:13.754000",
//     "confidence_score": 0.2,
//     "knowledge_source": "excel_qa",
//     "requires_human_referral": false
// }
}
