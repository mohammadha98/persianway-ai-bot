import { Routes } from '@angular/router';
import { AuthGuard } from './guards/auth.guard';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./pages/home/home.component').then(m => m.HomeComponent)
  },
  {
    path: 'chat',
    canActivate: [AuthGuard],
    loadComponent: () => import('./pages/chat/chat.component').then(m => m.ChatComponent)
  },
  {
    path: 'contribute',
    canActivate: [AuthGuard],
    loadComponent: () => import('./pages/contribute/contribute.component').then(m => m.ContributeComponent)
  },
  {
    path: 'settings',
    canActivate: [AuthGuard],
    loadComponent: () => import('./pages/settings/settings.component').then(m => m.SettingsComponent)
  },
  {
    path: 'llm-settings',
    canActivate: [AuthGuard],
    loadComponent: () => import('./pages/llm-settings/llm-settings.component').then(m => m.LlmSettingsComponent)
  },
  {
    path: 'rag-settings',
    canActivate: [AuthGuard],
    loadComponent: () => import('./pages/rag-settings/rag-settings.component').then(m => m.RagSettingsComponent)
  },
  {
    path: 'login',
    loadComponent: () => import('./pages/login/login.component').then(m => m.LoginComponent)
  },
  {
path:'users',
canActivate: [AuthGuard],
loadComponent: () => import('./pages/users/users.component').then(m => m.UsersComponent)
  },
  {
    path: 'knowledge',
    canActivate: [AuthGuard],
    loadComponent: () => import('./pages/knowledge/knowledge.component').then(m => m.KnowledgeComponent)
  },
  {
    path: '**',
    redirectTo: '/login'
  }
];
