import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

import { ToolsComponent } from './tools.component';
import { TavilySearchSettings, TavilyTestResponse } from '../../models/tavily-settings.model';
import { environment } from '../../../environments/environment';

/**
 * The web search page is where Tavily gets configured, so these tests drive it the
 * way an operator does: load the settings, edit them, save, test, reset.
 *
 * The point they pin down: the panel can set every field the API stores, and the
 * search stays restricted to `persianway.ir` unless more domains are typed in.
 */
describe('ToolsComponent', () => {
  let fixture: ComponentFixture<ToolsComponent>;
  let component: ToolsComponent;
  let httpMock: HttpTestingController;

  const base = `${environment.apiUrl}/api/config/tavily`;

  /** What a fresh backend document looks like after the site-search lock-down. */
  const storedSettings: TavilySearchSettings = {
    tavily_api_key: '****abcd',
    is_enabled: true,
    search_depth: 'advanced',
    max_results: 5,
    include_answer: true,
    include_domains: ['persianway.ir'],
    exclude_domains: [],
    snippet_length: 200
  };

  /** Creates the page and answers its initial GET with `settings`. */
  function createComponent(settings: Partial<TavilySearchSettings> = {}): void {
    fixture = TestBed.createComponent(ToolsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();

    httpMock.expectOne(req => req.url === base).flush({
      success: true,
      message: 'ok',
      settings: { ...storedSettings, ...settings }
    });
    fixture.detectChanges();
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [ToolsComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideNoopAnimations()
      ]
    });
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should create', () => {
    createComponent();
    expect(component).toBeTruthy();
  });

  it('loads every Tavily field the API stores into the form', () => {
    createComponent();

    expect(component.loading).toBeFalse();
    expect(component.form.value.search_depth).toBe('advanced');
    expect(component.form.value.max_results).toBe(5);
    expect(component.form.value.include_answer).toBeTrue();
    expect(component.form.value.snippet_length).toBe(200);
    // The mask the API answered with is what the field shows; it is never a key.
    expect(component.form.value.tavily_api_key).toBe('****abcd');
  });

  it('shows persianway.ir as the only allowed search domain', () => {
    createComponent();

    expect(component.includeDomains).toEqual(['persianway.ir']);
    // Nothing to save: the form already matches what the server stores.
    expect(component.hasChanges).toBeFalse();
  });

  it('saves the whole Tavily form back through the config router', () => {
    createComponent();

    component.form.patchValue({
      search_depth: 'basic',
      max_results: 3,
      include_answer: false,
      snippet_length: 120,
      include_domains_string: 'persianway.ir, blog.persianway.ir',
      exclude_domains_string: 'spam.example'
    });
    expect(component.hasChanges).toBeTrue();

    component.save();

    const request = httpMock.expectOne(base);
    expect(request.request.method).toBe('PUT');
    const body = request.request.body;
    expect(body.search_depth).toBe('basic');
    expect(body.max_results).toBe(3);
    expect(body.include_answer).toBeFalse();
    expect(body.snippet_length).toBe(120);
    expect(body.include_domains).toEqual(['persianway.ir', 'blog.persianway.ir']);
    expect(body.exclude_domains).toEqual(['spam.example']);
    // The untouched key is echoed as the mask, which the API reads as "keep it".
    expect(body.tavily_api_key).toBe('****abcd');

    request.flush({
      success: true,
      message: 'Tavily settings updated successfully',
      settings: { ...storedSettings, search_depth: 'basic', max_results: 3 }
    });

    expect(component.saving).toBeFalse();
    expect(component.saveMessageType).toBe('success');
    expect(component.hasChanges).toBeFalse();
  });

  it('refuses to save an emptied allowed-domains field', () => {
    createComponent();

    component.form.patchValue({ include_domains_string: '' });
    fixture.detectChanges();

    expect(component.form.get('include_domains_string')?.hasError('domainsRequired')).toBeTrue();
    expect(component.form.invalid).toBeTrue();

    component.save();
    // No request: the form never lets the search be opened to the whole web.
    httpMock.expectNone(base);
  });

  it('has a one-click way back to the persianway.ir-only list', () => {
    createComponent();

    component.form.patchValue({ include_domains_string: 'other.example' });
    component.usePersianWayOnly();

    expect(component.includeDomains).toEqual(['persianway.ir']);
  });

  it('runs the test search with the saved settings and shows the domain it used', () => {
    createComponent();

    component.testQuery = 'کود اوره';

    const response: TavilyTestResponse = {
      success: true,
      enabled: true,
      query: 'کود اوره',
      message: 'تست جستجو انجام شد',
      result: 'Sources:\n- [t](u): snippet...',
      elapsed_seconds: 1.2,
      include_domains: ['persianway.ir']
    };

    component.testSearch();

    const request = httpMock.expectOne(`${base}/test`);
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({ query: 'کود اوره' });
    request.flush(response);

    expect(component.testing).toBeFalse();
    expect(component.testResult?.result).toContain('Sources');
    expect(component.testResult?.include_domains).toEqual(['persianway.ir']);
    expect(component.testMessageType).toBe('success');
  });

  it('surfaces a failed test search instead of claiming success', () => {
    createComponent();

    component.testQuery = 'کود اوره';
    component.testSearch();
    httpMock.expectOne(`${base}/test`).flush({
      success: false,
      enabled: true,
      query: 'کود اوره',
      message: 'Error searching web: invalid api key',
      result: ''
    });

    expect(component.testMessageType).toBe('error');
    expect(component.testMessage).toContain('invalid api key');
  });

  it('ignores a test click without a query, so no request is sent', () => {
    createComponent();

    component.testQuery = '   ';
    component.testSearch();

    expect(component.testing).toBeFalse();
    httpMock.expectNone(`${base}/test`);
  });

  it('resets the web search section and reloads the defaults', () => {
    createComponent({ search_depth: 'basic', max_results: 1, include_domains: ['other.example'] });

    component.resetSettings();

    const request = httpMock.expectOne(`${base}/reset`);
    expect(request.request.method).toBe('POST');
    request.flush({
      success: true,
      message: 'Tavily settings reset to defaults',
      settings: { ...storedSettings, tavily_api_key: '****efgh' }
    });
    fixture.detectChanges();

    expect(component.resetting).toBeFalse();
    expect(component.form.value.search_depth).toBe('advanced');
    expect(component.includeDomains).toEqual(['persianway.ir']);
    expect(component.form.value.tavily_api_key).toBe('****efgh');
  });

  it('shows the server detail when a save is rejected', () => {
    createComponent();

    component.form.patchValue({ max_results: 9 });
    component.save();
    httpMock.expectOne(base).flush(
      { detail: 'Invalid Tavily settings - max_results: out of range' },
      { status: 400, statusText: 'Bad Request' }
    );

    expect(component.saveMessageType).toBe('error');
    expect(component.saveMessage).toContain('Invalid Tavily settings');
  });
});
