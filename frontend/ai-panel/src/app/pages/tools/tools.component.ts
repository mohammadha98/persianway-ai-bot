import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import {
  AbstractControl,
  FormBuilder,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
  ValidationErrors,
  Validators
} from '@angular/forms';
import { WebToolService } from '../../services/web-tool.service';
import { TavilySearchSettings, TavilyTestResponse } from '../../models/tavily-settings.model';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatButtonModule } from '@angular/material/button';
import { MatIcon } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatDivider } from '@angular/material/divider';

/**
 * Web search is a *site* search: Tavily has to stay restricted to at least one
 * domain, otherwise an emptied field would silently hand the bot's API key the
 * whole internet (the API rejects the same payload with a 400).
 */
function atLeastOneDomain(control: AbstractControl): ValidationErrors | null {
  const domains = String(control.value || '')
    .split(',')
    .map(value => value.trim())
    .filter(value => value.length > 0);

  return domains.length > 0 ? null : { domainsRequired: true };
}

@Component({
  selector: 'app-tools',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
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
  testing = false;
  resetting = false;

  saveMessage = '';
  saveMessageType: 'success' | 'error' = 'success';

  testQuery = '';
  testResult: TavilyTestResponse | null = null;
  testMessage = '';
  testMessageType: 'success' | 'error' = 'success';

  /** Serialized copy of the settings as last read from / written to the server. */
  private savedSnapshot = '';

  constructor(private fb: FormBuilder, private webTool: WebToolService) {
    this.form = this.fb.group({
      tavily_api_key: [null],
      is_enabled: [true],
      search_depth: ['advanced', Validators.required],
      max_results: [5, [Validators.required, Validators.min(1), Validators.max(20)]],
      include_answer: [true],
      // Pre-filled with the locked default, and required to stay non-empty: the
      // only site Tavily may read from is `persianway.ir` unless an operator adds
      // more domains on purpose.
      include_domains_string: ['persianway.ir', atLeastOneDomain],
      exclude_domains_string: [''],
      snippet_length: [200, [Validators.required, Validators.min(1)]]
    });
  }

  /** True only when the form differs from what the server currently stores. */
  get hasChanges(): boolean {
    return this.serialize(this.buildPayload()) !== this.savedSnapshot;
  }

  /** The PUT body as the form currently stands (domains normalized to arrays). */
  private buildPayload(): Partial<TavilySearchSettings> {
    return {
      tavily_api_key: this.form.value.tavily_api_key,
      is_enabled: this.form.value.is_enabled,
      search_depth: this.form.value.search_depth,
      max_results: this.form.value.max_results,
      include_answer: this.form.value.include_answer,
      snippet_length: this.form.value.snippet_length,
      include_domains: this.splitDomains(this.form.value.include_domains_string),
      exclude_domains: this.splitDomains(this.form.value.exclude_domains_string)
    };
  }

  /**
   * Canonical string for a payload: a fixed field order, so two payloads that
   * describe the same settings always compare equal whatever order they were
   * built in.
   */
  private serialize(payload: Partial<TavilySearchSettings>): string {
    return JSON.stringify([
      payload.tavily_api_key ?? null,
      payload.is_enabled ?? null,
      payload.search_depth ?? null,
      payload.max_results ?? null,
      payload.include_answer ?? null,
      payload.snippet_length ?? null,
      payload.include_domains ?? [],
      payload.exclude_domains ?? []
    ]);
  }

  get searchEnabled(): boolean {
    return !!this.form.value.is_enabled;
  }

  /** The domains the form currently allows; the field is required to hold some. */
  get includeDomains(): string[] {
    return this.splitDomains(this.form.value.include_domains_string);
  }

  get busy(): boolean {
    return this.loading || this.saving || this.testing || this.resetting;
  }

  ngOnInit(): void {
    this.loading = true;
    this.webTool.getTavilySettings().subscribe({
      next: (res) => {
        this.applySettings(res.settings);
        this.loading = false;
      },
      error: (err: HttpErrorResponse) => {
        this.loading = false;
        this.saveMessageType = 'error';
        this.saveMessage = this.describeError(err, 'دریافت تنظیمات جستجو ناموفق بود.');
      }
    });
  }

  /** Writes server settings into the form and records them as the saved state. */
  private applySettings(settings: TavilySearchSettings): void {
    this.form.patchValue({
      tavily_api_key: settings.tavily_api_key,
      is_enabled: settings.is_enabled,
      search_depth: settings.search_depth,
      max_results: settings.max_results,
      include_answer: settings.include_answer,
      include_domains_string: (settings.include_domains || []).join(', '),
      exclude_domains_string: (settings.exclude_domains || []).join(', '),
      snippet_length: settings.snippet_length
    });
    this.savedSnapshot = this.serialize(this.buildPayload());
  }

  private splitDomains(value: string): string[] {
    return (value || '')
      .split(',')
      .map(v => v.trim())
      .filter(v => v.length > 0);
  }

  /** Locks the allowed-domains field back to the one site search may read from. */
  usePersianWayOnly(): void {
    this.form.patchValue({ include_domains_string: 'persianway.ir' });
  }

  save(): void {
    if (this.form.invalid || this.busy) return;
    this.saving = true;
    this.saveMessage = '';
    const payload = this.buildPayload();
    this.webTool.updateTavilySettings(payload).subscribe({
      next: (res) => {
        this.saving = false;
        this.saveMessageType = 'success';
        this.saveMessage = res.message || 'Saved';
        // The response is what the server actually stored (normalised domains, a
        // masked key), so it — not the form — becomes the new baseline.
        this.applySettings(res.settings || (payload as TavilySearchSettings));
      },
      error: (err: HttpErrorResponse) => {
        this.saving = false;
        this.saveMessageType = 'error';
        this.saveMessage = this.describeError(err, 'ذخیره تنظیمات ناموفق بود.');
      }
    });
  }

  /**
   * Runs one real search with the settings already stored on the server, so an
   * operator can prove the key, the allowed domains and the search depth work
   * before a live question depends on them.
   */
  testSearch(): void {
    const query = (this.testQuery || '').trim();
    if (!query || this.busy) return;

    this.testing = true;
    this.testResult = null;
    this.testMessage = '';

    this.webTool.testTavilySearch(query).subscribe({
      next: (res: TavilyTestResponse) => {
        this.testing = false;
        this.testResult = res;
        this.testMessageType = res.success ? 'success' : 'error';
        this.testMessage = res.message
          || (res.success ? 'تست جستجو با موفقیت انجام شد' : 'تست جستجو نتیجه‌ای نداشت');
      },
      error: (err: HttpErrorResponse) => {
        this.testing = false;
        this.testMessageType = 'error';
        this.testMessage = this.describeError(err, 'اجرای تست جستجو ناموفق بود.');
      }
    });
  }

  /** Restores the web search section only; the LLM/RAG settings stay untouched. */
  resetSettings(): void {
    if (this.busy) return;

    this.resetting = true;
    this.saveMessage = '';

    this.webTool.resetTavilySettings().subscribe({
      next: (res) => {
        this.resetting = false;
        this.saveMessageType = 'success';
        this.saveMessage = res.message || 'تنظیمات جستجو بازنشانی شد';
        this.applySettings(res.settings);
        // The last test ran against the settings that were just discarded.
        this.testResult = null;
        this.testMessage = '';
      },
      error: (err: HttpErrorResponse) => {
        this.resetting = false;
        this.saveMessageType = 'error';
        this.saveMessage = this.describeError(err, 'بازنشانی تنظیمات جستجو ناموفق بود.');
      }
    });
  }

  /** The server's message when it sent one: validation errors are the operator's clue. */
  private describeError(error: HttpErrorResponse, fallback: string): string {
    const detail = error?.error?.detail;
    if (typeof detail === 'string' && detail.trim()) {
      return detail;
    }
    return fallback;
  }
}
