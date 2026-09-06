import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatDialogModule, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTabsModule } from '@angular/material/tabs';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatExpansionModule } from '@angular/material/expansion';
import { ClipboardModule } from '@angular/cdk/clipboard';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';

export interface ContextDialogData {
  data: any;
  isLoading: boolean;
}

@Component({
  selector: 'app-context-dialog',
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
    MatExpansionModule,
    ClipboardModule,
    MatSnackBarModule
  ],
  templateUrl: './context-dialog.component.html',
  styleUrl: './context-dialog.component.scss'
})
export class ContextDialogComponent {
  selectedTabIndex = 0;

  constructor(
    public dialogRef: MatDialogRef<ContextDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: ContextDialogData,
    private snackBar: MatSnackBar
  ) {}

  onClose(): void {
    this.dialogRef.close();
  }

  copyToClipboard(text: string, label: string): void {
    this.snackBar.open(`${label} کپی شد`, 'بستن', {
      duration: 2000,
      horizontalPosition: 'center',
      verticalPosition: 'bottom'
    });
  }

  getRoleLabel(role: string): string {
    return role === 'user' ? 'کاربر' : 'دستیار';
  }
}