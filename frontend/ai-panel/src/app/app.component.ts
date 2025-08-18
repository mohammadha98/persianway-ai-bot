import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterModule, Router, NavigationEnd } from '@angular/router';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatMenuModule } from '@angular/material/menu';
import { MatDividerModule } from '@angular/material/divider';
import { HttpClientModule } from '@angular/common/http';
import { AuthService, User, PermissionType } from './services/auth.service';
import { Observable } from 'rxjs';
import { filter } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, 
    RouterOutlet, 
    RouterModule,
    MatToolbarModule, 
    MatButtonModule, 
    MatIconModule,
    MatMenuModule,
    MatDividerModule,
    HttpClientModule
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements OnInit {
  title = 'Persianway AI';
  isAuthenticated$: Observable<boolean>;
  currentUser$: Observable<User | null>;
  currentUser: User | null = null;
  isOnChatRoute = false;
  // Make PermissionType available in the template
  PermissionType = PermissionType;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {
    this.isAuthenticated$ = this.authService.isAuthenticated$;
    this.currentUser$ = this.authService.currentUser$;
  }

  ngOnInit(): void {
    this.currentUser$.subscribe(user => {
      this.currentUser = user;
    });

    // Listen for route changes to hide footer on chat page
    this.router.events.pipe(
      filter(event => event instanceof NavigationEnd)
    ).subscribe((event: NavigationEnd) => {
      this.isOnChatRoute = event.url === '/chat' || event.url.startsWith('/chat/');
    });

    // Check initial route
    this.isOnChatRoute = this.router.url === '/chat' || this.router.url.startsWith('/chat/');
  }

  logout(): void {
    this.authService.logout();
  }

  navigateToLogin(): void {
    this.router.navigate(['/login']);
  }

  hasRole(role: string): boolean {
    return this.authService.hasRole(role);
  }

  hasPermission(permissionType: string | PermissionType): boolean {
    // If user is admin, they have access to everything
    if (this.hasRole('admin')) {
      return true;
    }
    
    // Otherwise check specific permission
    return this.authService.hasPermission(permissionType as PermissionType);
  }
}
