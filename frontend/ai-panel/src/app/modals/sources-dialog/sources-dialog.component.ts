import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatDialogModule, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTabsModule } from '@angular/material/tabs';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ClipboardModule } from '@angular/cdk/clipboard';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';

export interface SourcesDialogData {
  sources: any[];
  isLoading: boolean;
}

@Component({
  selector: 'app-sources-dialog',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatDialogModule,
    MatProgressSpinnerModule,
    MatTabsModule,
    MatChipsModule,
    MatTooltipModule,
    ClipboardModule,
    MatSnackBarModule
  ],
  templateUrl: './sources-dialog.component.html',
  styleUrl: './sources-dialog.component.scss'
})
export class SourcesDialogComponent {
  constructor(
    public dialogRef: MatDialogRef<SourcesDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: SourcesDialogData,
    private snackBar: MatSnackBar
  ) {}

  onClose(): void {
    this.dialogRef.close();
  }

  copyContent(content: string, sourceTitle: string): void {
    this.snackBar.open(`محتوای "${sourceTitle}" کپی شد`, 'بستن', {
      duration: 2000,
      horizontalPosition: 'center',
      verticalPosition: 'bottom'
    });
  }

  getSourceTypeLabel(type: string): string {
    const labels: { [key: string]: string } = {
      'pdf': 'PDF',
      'qa': 'پرسش و پاسخ',
      'docx': 'Word',
      'excel': 'اکسل',
      'text': 'متن'
    };
    return labels[type] || type;
  }

  getRelevanceColor(score: number): string {
    if (score >= 0.8) return '#4caf50';
    if (score >= 0.6) return '#ff9800';
    return '#f44336';
  }

  getRelevanceLabel(score: number): string {
    if (score >= 0.8) return 'بسیار مرتبط';
    if (score >= 0.6) return 'مرتبط';
    return 'کم مرتبط';
  }
}