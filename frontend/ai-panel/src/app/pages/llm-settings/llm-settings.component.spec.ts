import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LlmSettingsComponent } from './llm-settings.component';

describe('LlmSettingsComponent', () => {
  let component: LlmSettingsComponent;
  let fixture: ComponentFixture<LlmSettingsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LlmSettingsComponent]
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
