import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Subject, of } from 'rxjs';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

import { ChatComponent } from './chat.component';
import { ChatService, EnhancedChatRequest } from '../../services/chat.service';
import { AuthService } from '../../services/auth.service';

/**
 * Failure handling of the chat page.
 *
 * When the stream dies without delivering an answer the user must not be left at a
 * dead end: the bubble is marked as failed and gets a retry button. These tests pin
 * down that the retry re-sends the *same* question through the service, that it
 * reuses the failed bubble instead of stacking another one, and that a delivered
 * answer clears the failure state.
 */
describe('ChatComponent', () => {
  let fixture: ComponentFixture<ChatComponent>;
  let component: ChatComponent;
  let chatService: {
    requests: EnhancedChatRequest[];
    streams: Subject<any>[];
    prepareChatRequest: jasmine.Spy;
    streamChatMessage: jasmine.Spy;
    getUserFromStorage: jasmine.Spy;
    generateUserIdFromEmail: jasmine.Spy;
    getConversationsByEmail: jasmine.Spy;
    resetSession: jasmine.Spy;
  };

  function createComponent(): void {
    fixture = TestBed.createComponent(ChatComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  /** Asks a question the way the input box does: user bubble + open stream. */
  function ask(question: string): void {
    component.currentMessage = question;
    component.sendMessage();
  }

  /**
   * The bubble the component is filling right now.
   *
   * The answer bubble is created lazily on the first stream event, so this has to
   * be read *after* one arrived.
   */
  function answerBubble() {
    return component.messages[component.messages.length - 1];
  }

  beforeEach(() => {
    const streams: Subject<any>[] = [];
    const requests: EnhancedChatRequest[] = [];

    chatService = {
      streams,
      requests,
      prepareChatRequest: jasmine.createSpy('prepareChatRequest').and.callFake((message: string) => {
        const request: EnhancedChatRequest = { user_id: 'user-1', message, session_id: 'session-1' };
        requests.push(request);
        return request;
      }),
      streamChatMessage: jasmine.createSpy('streamChatMessage').and.callFake(() => {
        const stream = new Subject<any>();
        streams.push(stream);
        return stream.asObservable();
      }),
      getUserFromStorage: jasmine.createSpy('getUserFromStorage')
        .and.returnValue({ email: 'operator@persianway.ir' }),
      generateUserIdFromEmail: jasmine.createSpy('generateUserIdFromEmail').and.returnValue('user-1'),
      getConversationsByEmail: jasmine.createSpy('getConversationsByEmail')
        .and.returnValue(of({ conversations: [] })),
      resetSession: jasmine.createSpy('resetSession')
    };

    TestBed.configureTestingModule({
      imports: [ChatComponent],
      providers: [
        provideNoopAnimations(),
        { provide: ChatService, useValue: chatService },
        { provide: AuthService, useValue: { hasRole: () => true, hasPermission: () => true } }
      ]
    });
  });

  it('should create', () => {
    createComponent();
    expect(component).toBeTruthy();
  });

  it('marks the answer as failed and keeps the question for a retry', () => {
    createComponent();
    ask('قیمت کود اوره چند است؟');

    expect(chatService.streamChatMessage).toHaveBeenCalledTimes(1);
    expect(chatService.requests[0].message).toBe('قیمت کود اوره چند است؟');

    chatService.streams[0].error({ message: 'network down', code: 'network_error' });
    fixture.detectChanges();

    const answer = answerBubble();
    // The user bubble and the failed answer: no extra error bubble was pushed.
    expect(component.messages.length).toBe(2);
    expect(answer.isUser).toBeFalse();
    expect(answer.failed).toBeTrue();
    expect(answer.retryQuestion).toBe('قیمت کود اوره چند است؟');
    expect(answer.content).toBe('network down');
    expect(component.isLoading).toBeFalse();
  });

  it('re-sends the same question on retry and reuses the failed bubble', () => {
    createComponent();
    ask('قیمت کود اوره چند است؟');
    chatService.streams[0].error({ message: 'network down', code: 'network_error' });
    const answer = answerBubble();

    component.retryMessage(answer);

    expect(chatService.streamChatMessage).toHaveBeenCalledTimes(2);
    expect(chatService.requests[1].message).toBe('قیمت کود اوره چند است؟');
    expect(answer.retrying).toBeTrue();
    expect(answer.failed).toBeFalse();

    chatService.streams[1].next({
      type: 'done',
      answer: 'هر کیسه ۴۵۰ هزار تومان است.',
      message_id: 'm-1',
      conversation_id: 'c-1',
      session_id: 'session-1',
      truncated: false
    });
    fixture.detectChanges();

    expect(component.messages.length).toBe(2);
    expect(answer.content).toBe('هر کیسه ۴۵۰ هزار تومان است.');
    expect(answer.failed).toBeFalse();
    expect(answer.retrying).toBeFalse();
    expect(answer.retryQuestion).toBeUndefined();
    expect(component.isLoading).toBeFalse();
  });

  it('keeps a partial answer on screen and still offers the retry', () => {
    createComponent();
    ask('قیمت کود اوره چند است؟');

    chatService.streams[0].next({ type: 'chunk', content: 'نیمه‌ی پاسخ' });
    chatService.streams[0].error({ message: 'stream broke', code: 'stream_read_error' });

    const answer = answerBubble();
    expect(answer.content).toBe('نیمه‌ی پاسخ');
    expect(answer.failed).toBeTrue();
    expect(answer.retryQuestion).toBe('قیمت کود اوره چند است؟');
  });

  it('leaves a delivered answer without any retry state', () => {
    createComponent();
    ask('قیمت کود اوره چند است؟');

    chatService.streams[0].next({
      type: 'done',
      answer: 'پاسخ کامل',
      message_id: 'm-2',
      conversation_id: 'c-2',
      session_id: 'session-1',
      truncated: false
    });

    const answer = answerBubble();
    expect(answer.failed).toBeFalsy();
    expect(answer.retryQuestion).toBeUndefined();
    expect(component.messages.length).toBe(2);
  });

  it('ignores a retry while another stream is running', () => {
    createComponent();
    ask('قیمت کود اوره چند است؟');
    chatService.streams[0].error({ message: 'network down', code: 'network_error' });
    const answer = answerBubble();

    component.isLoading = true;
    component.retryMessage(answer);

    expect(chatService.streamChatMessage).toHaveBeenCalledTimes(1);
    expect(answer.failed).toBeTrue();
  });

  it('ignores a retry of an answer that was never asked or never failed', () => {
    createComponent();

    component.retryMessage({
      id: 'x',
      content: 'پاسخ سالم',
      isUser: false,
      timestamp: new Date()
    });

    expect(chatService.streamChatMessage).not.toHaveBeenCalled();
  });

  it('renders a retry button for a failed answer and retries when it is clicked', () => {
    createComponent();
    ask('قیمت کود اوره چند است؟');
    chatService.streams[0].error({ message: 'network down', code: 'network_error' });
    fixture.detectChanges();

    const buttons = fixture.nativeElement.querySelectorAll('.retry-row .retry-btn');
    expect(buttons.length).toBe(1);

    (buttons[0] as HTMLButtonElement).click();
    fixture.detectChanges();

    // The click re-sent the same question, and no second bubble was added.
    expect(chatService.streamChatMessage).toHaveBeenCalledTimes(2);
    expect(chatService.requests[1].message).toBe('قیمت کود اوره چند است؟');
    expect(component.messages.length).toBe(2);

    chatService.streams[1].next({
      type: 'done',
      answer: 'پاسخ کامل',
      message_id: 'm-3',
      conversation_id: 'c-3',
      session_id: 'session-1',
      truncated: false
    });
    fixture.detectChanges();

    // Once answered, the retry affordance is gone.
    expect(fixture.nativeElement.querySelectorAll('.retry-row').length).toBe(0);
  });

  it('ignores a second question while a stream is still running', () => {
    createComponent();
    ask('قیمت کود اوره چند است؟');

    component.currentMessage = 'سوال دوم';
    component.sendMessage();

    // Only the first question was sent, and no second user bubble was added.
    expect(chatService.streamChatMessage).toHaveBeenCalledTimes(1);
    expect(component.messages.length).toBe(1);
  });
});
