import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatDialogModule } from '@angular/material/dialog';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatChipsModule } from '@angular/material/chips';

export interface KnowledgeItem {
  hash_id: string;
  title: string;
  content: string;
  category: string;
  created_at: string;
  updated_at: string;
  synced?: boolean | null;
}

@Component({
  selector: 'app-knowledge-details-modal',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatIconModule,
    MatButtonModule,
    MatDialogModule,
    MatTooltipModule,
    MatChipsModule
  ],
  templateUrl: './knowledge-details-modal.component.html',
  styleUrl: './knowledge-details-modal.component.scss'
})
export class KnowledgeDetailsModalComponent {
  knowledge: KnowledgeItem;

  constructor(
    public dialogRef: MatDialogRef<KnowledgeDetailsModalComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { knowledge: KnowledgeItem }
  ) {
    this.knowledge = data.knowledge;
  }

  onClose(): void {
    this.dialogRef.close();
  }

  getSyncStatusText(): string {
    if (this.knowledge.synced === true) {
      return 'همگام‌سازی شده';
    } else if (this.knowledge.synced === false) {
      return 'همگام‌سازی نشده';
    } else {
      return 'نامشخص';
    }
  }

  getSyncStatusIcon(): string {
    if (this.knowledge.synced === true) {
      return 'check_circle';
    } else if (this.knowledge.synced === false) {
      return 'cancel';
    } else {
      return 'help';
    }
  }

  getSyncStatusClass(): string {
    if (this.knowledge.synced === true) {
      return 'synced';
    } else if (this.knowledge.synced === false) {
      return 'not-synced';
    } else {
      return 'unknown';
    }
  }
}