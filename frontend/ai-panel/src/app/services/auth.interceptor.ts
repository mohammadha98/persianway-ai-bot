import { Injectable } from '@angular/core';
import {
  HttpErrorResponse,
  HttpRequest,
  HttpHandler,
  HttpEvent,
  HttpInterceptor
} from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { AuthService } from './auth.service';

@Injectable()
export class AuthInterceptor implements HttpInterceptor {

  constructor(private authService: AuthService) {}

  intercept(request: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
    const authToken = this.authService.getToken();
    if (authToken) {
      request = request.clone({
        setHeaders: {
          Authorization: `Bearer ${authToken}`
        }
      });
    }

    const sentToken = request.headers.has('Authorization');

    return next.handle(request).pipe(
      catchError((error: HttpErrorResponse) => {
        // 401 means the token has expired or was rejected, or the account was
        // disabled -- every one of those ends the session, so the operator is
        // sent to the login page instead of being left on a panel whose calls
        // all fail. The login request itself is excluded: a wrong password is
        // a 401 too and must keep its own inline message.
        const tokenRejected = error.status === 401;
        // A request that carried no token answers 403 (HTTPBearer) rather than
        // 401. If this tab still holds a user in memory, its session was wiped
        // elsewhere -- another tab signed out, storage was cleared -- and it is
        // just as dead as an expired token.
        const sessionVanished =
          error.status === 403 && !sentToken && !!this.authService.getCurrentUser();

        if ((tokenRejected || sessionVanished) && !request.url.includes('/api/users/login')) {
          this.authService.expireSession();
        }
        return throwError(() => error);
      })
    );
  }
}
