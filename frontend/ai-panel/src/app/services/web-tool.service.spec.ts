import { TestBed } from '@angular/core/testing';

import { WebToolService } from './web-tool.service';

describe('WebToolService', () => {
  let service: WebToolService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(WebToolService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
