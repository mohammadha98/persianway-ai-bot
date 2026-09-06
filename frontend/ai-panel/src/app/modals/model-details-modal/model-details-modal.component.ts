import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatDialogRef, MAT_DIALOG_DATA, MatDialogModule } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatDividerModule } from '@angular/material/divider';
import { OpenRouterModel } from '../../services/models.service';

@Component({
  selector: 'app-model-details-modal',
  standalone: true,
  imports: [
    CommonModule,
    MatDialogModule,
    MatButtonModule,
    MatIconModule,
    MatCardModule,
    MatChipsModule,
    MatDividerModule
  ],
  templateUrl: './model-details-modal.component.html',
  styleUrl: './model-details-modal.component.scss'
})
export class ModelDetailsModalComponent {
  constructor(
    public dialogRef: MatDialogRef<ModelDetailsModalComponent>,
    @Inject(MAT_DIALOG_DATA) public model: OpenRouterModel
  ) {}

  onClose(): void {
    this.dialogRef.close();
  }

  formatPrice(price: string): string {
    const numPrice = parseFloat(price);
    if (numPrice === 0) return 'رایگان';
    if (numPrice < 0.001) return `$${(numPrice * 1000000).toFixed(2)}/1M tokens`;
    if (numPrice < 1) return `$${(numPrice * 1000).toFixed(2)}/1K tokens`;
    return `$${numPrice.toFixed(4)}/token`;
  }

  formatContextLength(length: number | null): string {
    if (!length) return 'نامشخص';
    if (length >= 1000000) return `${(length / 1000000).toFixed(1)}M`;
    if (length >= 1000) return `${(length / 1000).toFixed(0)}K`;
    return length.toString();
  }

  formatDate(timestamp: number): string {
    return new Date(timestamp * 1000).toLocaleDateString('fa-IR');
  }

  copyModelId(): void {
    navigator.clipboard.writeText(this.model.id);
  }
}