import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';

import { WebToolService } from './web-tool.service';
import { TavilySearchSettings } from '../models/tavily-settings.model';
import { environment } from '../../environments/environment';

/**
 * The panel's half of the `/api/config/tavily` contract: which URL and verb each
 * button uses, and that the web search settings always come from the config router
 * (never from a cache).
 */
describe('WebToolService', () => {
  let service: WebToolService;
  let httpMock: HttpTestingController;
  const base = `${environment.apiUrl}/api/config/tavily`;

  const settings: TavilySearchSettings = {
    tavily_api_key: '****-key',
    is_enabled: true,
    search_depth: 'advanced',
    max_results: 5,
    include_answer: true,
    include_domains: ['persianway.ir'],
    exclude_domains: [],
    snippet_length: 200
  };

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()]
    });
    service = TestBed.inject(WebToolService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('reads the settings from /api/config/tavily without caching', () => {
    let received: TavilySearchSettings | undefined;
    service.getTavilySettings().subscribe(res => (received = res.settings));

    const request = httpMock.expectOne(req => req.url === base);
    expect(request.request.method).toBe('GET');
    expect(request.request.headers.get('Cache-Control')).toBe('no-cache');
    expect(request.request.headers.get('Pragma')).toBe('no-cache');
    // Cache busting: a proxy must never answer with an older document.
    expect(request.request.params.has('_t')).toBeTrue();

    request.flush({ success: true, message: 'ok', settings });

    expect(received?.include_domains).toEqual(['persianway.ir']);
  });

  it('writes the settings with PUT, sending only the fields the panel owns', () => {
    service.updateTavilySettings({ include_domains: ['persianway.ir'], is_enabled: false })
      .subscribe();

    const request = httpMock.expectOne(base);
    expect(request.request.method).toBe('PUT');
    expect(request.request.body).toEqual({
      include_domains: ['persianway.ir'],
      is_enabled: false
    });
    request.flush({ success: true, message: 'saved', settings });
  });

  it('runs the test search through POST /api/config/tavily/test', () => {
    service.testTavilySearch('کود اوره').subscribe();

    const request = httpMock.expectOne(`${base}/test`);
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({ query: 'کود اوره' });
    request.flush({
      success: true,
      enabled: true,
      query: 'کود اوره',
      message: 'ok',
      result: 'Sources: ...',
      include_domains: ['persianway.ir']
    });
  });

  it('resets only the web search section through POST /api/config/tavily/reset', () => {
    service.resetTavilySettings().subscribe();

    const request = httpMock.expectOne(`${base}/reset`);
    expect(request.request.method).toBe('POST');
    request.flush({ success: true, message: 'reset', settings });
  });
});
