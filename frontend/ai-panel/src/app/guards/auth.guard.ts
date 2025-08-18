import { Injectable } from '@angular/core';
import { CanActivate, Router, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { Observable } from 'rxjs';
import { map, take } from 'rxjs/operators';
import { AuthService, PermissionType } from '../services/auth.service';

@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {
  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  canActivate(
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot
  ): Observable<boolean> | Promise<boolean> | boolean {
    return this.authService.isAuthenticated$.pipe(
      take(1),
      map(isAuthenticated => {
        if (!isAuthenticated) {
          // Store the attempted URL for redirecting after login
          this.router.navigate(['/login'], {
            queryParams: { returnUrl: state.url }
          });
          return false;
        }
        
        // If user is admin, allow access to all routes
        if (this.authService.hasRole('admin')) {
          return true;
        }
        
        // Check permissions based on route
        const url = state.url;
        
        // Define which permission is required for each route
        if (url.startsWith('/chat')) {
          return this.authService.hasPermission(PermissionType.Chat);
        } else if (url.startsWith('/contribute')) {
          return this.authService.hasPermission(PermissionType.Contribute);
        } else if (url.startsWith('/settings')) {
          return this.authService.hasPermission(PermissionType.Settings);
        } else if (url.startsWith('/knowledge')) {
          return this.authService.hasPermission(PermissionType.Guide);
        } else if (url.startsWith('/conversation-search')) {
          return this.authService.hasPermission(PermissionType.Conversations);
        } else if (url.startsWith('/llm-settings')) {
          return this.authService.hasPermission(PermissionType.LLM);
        } else if (url.startsWith('/rag-settings')) {
          return this.authService.hasPermission(PermissionType.RAG);
        } else if (url.startsWith('/docs')) {
          return this.authService.hasPermission(PermissionType.Docs);
        }
        
        // For home page or other routes without specific permissions
        return true;
      })
    );
  }
}