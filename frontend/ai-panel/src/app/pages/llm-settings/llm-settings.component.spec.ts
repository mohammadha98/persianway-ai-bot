import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

import { LlmSettingsComponent } from './llm-settings.component';

describe('LlmSettingsComponent', () => {
  let component: LlmSettingsComponent;
  let fixture: ComponentFixture<LlmSettingsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LlmSettingsComponent],
      // The page loads the LLM settings from the API on init.
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideNoopAnimations()
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LlmSettingsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
