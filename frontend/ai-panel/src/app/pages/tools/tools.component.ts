import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { WebToolService } from '../../services/web-tool.service';
import { TavilySearchSettings } from '../../models/tavily-settings.model';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatButtonModule } from '@angular/material/button';
import { MatIcon } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatDivider } from '@angular/material/divider';

@Component({
  selector: 'app-tools',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatSlideToggleModule,
    MatButtonModule,
    MatIcon,
    MatProgressSpinnerModule,
    MatDivider
  ],
  templateUrl: './tools.component.html',
  styleUrl: './tools.component.scss'
})
export class ToolsComponent implements OnInit {
  form: FormGroup;
  loading = false;
  saving = false;
  saveMessage = '';

  constructor(private fb: FormBuilder, private webTool: WebToolService) {
    this.form = this.fb.group({
      tavily_api_key: [null],
      is_enabled: [true],
      search_depth: ['advanced', Validators.required],
      max_results: [5, [Validators.required, Validators.min(1), Validators.max(20)]],
      include_answer: [true],
      include_domains_string: [''],
      exclude_domains_string: [''],
      snippet_length: [200, [Validators.required, Validators.min(1)]]
    });
  }

  ngOnInit(): void {
    this.loading = true;
    this.webTool.getTavilySettings().subscribe({
      next: (res) => {
        const s: TavilySearchSettings = res.settings;
        this.form.patchValue({
          tavily_api_key: s.tavily_api_key,
          is_enabled: s.is_enabled,
          search_depth: s.search_depth,
          max_results: s.max_results,
          include_answer: s.include_answer,
          include_domains_string: (s.include_domains || []).join(', '),
          exclude_domains_string: (s.exclude_domains || []).join(', '),
          snippet_length: s.snippet_length
        });
        this.loading = false;
      },
      error: () => {
        this.loading = false;
      }
    });
  }

  private splitDomains(value: string): string[] {
    return (value || '')
      .split(',')
      .map(v => v.trim())
      .filter(v => v.length > 0);
  }

  save(): void {
    if (this.form.invalid) return;
    this.saving = true;
    this.saveMessage = '';
    const include_domains = this.splitDomains(this.form.value.include_domains_string);
    const exclude_domains = this.splitDomains(this.form.value.exclude_domains_string);
    const payload: Partial<TavilySearchSettings> = {
      tavily_api_key: this.form.value.tavily_api_key,
      is_enabled: this.form.value.is_enabled,
      search_depth: this.form.value.search_depth,
      max_results: this.form.value.max_results,
      include_answer: this.form.value.include_answer,
      snippet_length: this.form.value.snippet_length,
      include_domains,
      exclude_domains
    };
    this.webTool.updateTavilySettings(payload).subscribe({
      next: (res) => {
        this.saving = false;
        this.saveMessage = res.message || 'Saved';
      },
      error: (err) => {
        this.saving = false;
        this.saveMessage = 'Failed to save';
      }
    });
  }
}
