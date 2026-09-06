import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { map, tap } from 'rxjs/operators';
import { environment } from '../../environments/environment';

// OpenRouter API Response Interfaces
export interface ModelArchitecture {
  input_modalities: string[];
  output_modalities: string[];
  tokenizer: string;
  instruct_type: string;
}

export interface TopProvider {
  is_moderated: boolean;
  context_length: number;
  max_completion_tokens: number;
}

export interface ModelPricing {
  prompt: string;
  completion: string;
  image: string;
  request: string;
  web_search: string;
  internal_reasoning: string;
  input_cache_read: string;
  input_cache_write: string;
}

export interface OpenRouterModel {
  id: string;
  name: string;
  created: number;
  description: string;
  architecture: ModelArchitecture;
  top_provider: TopProvider;
  pricing: ModelPricing;
  canonical_slug: string | null;
  context_length: number | null;
  hugging_face_id: string | null;
  per_request_limits: any;
  supported_parameters: string[] | null;
}

export interface OpenRouterModelsResponse {
  data: OpenRouterModel[];
}

export interface ModelQueryParams {
  category?: string;
  use_rss?: boolean;
  use_rss_chat_links?: boolean;
}

export interface PaginatedModelsResponse {
  data: OpenRouterModel[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

interface CachedModelsData {
  data: OpenRouterModel[];
  timestamp: number;
  expiry: number;
}

@Injectable({
  providedIn: 'root'
})
export class ModelsService {
  private openRouterApiUrl = 'https://openrouter.ai/api/v1';
  private apiUrl = environment.apiUrl || 'http://localhost:8000';
  private cacheKey = 'openrouter_models_cache';
  private cacheExpiryHours = 1; // Cache for 1 hour
  private allModels: OpenRouterModel[] = [];

  constructor(private http: HttpClient) { }

  /**
   * Check if cached data is valid
   */
  private isCacheValid(): boolean {
    try {
      const cached = localStorage.getItem(this.cacheKey);
      if (!cached) return false;
      
      const cachedData: CachedModelsData = JSON.parse(cached);
      return Date.now() < cachedData.expiry;
    } catch {
      return false;
    }
  }

  /**
   * Get cached models data
   */
  private getCachedModels(): OpenRouterModel[] | null {
    try {
      const cached = localStorage.getItem(this.cacheKey);
      if (!cached) return null;
      
      const cachedData: CachedModelsData = JSON.parse(cached);
      if (Date.now() >= cachedData.expiry) {
        localStorage.removeItem(this.cacheKey);
        return null;
      }
      
      return cachedData.data;
    } catch {
      localStorage.removeItem(this.cacheKey);
      return null;
    }
  }

  /**
   * Cache models data
   */
  private cacheModels(models: OpenRouterModel[]): void {
    try {
      const cacheData: CachedModelsData = {
        data: models,
        timestamp: Date.now(),
        expiry: Date.now() + (this.cacheExpiryHours * 60 * 60 * 1000)
      };
      localStorage.setItem(this.cacheKey, JSON.stringify(cacheData));
      this.allModels = models;
    } catch (error) {
      console.warn('Failed to cache models data:', error);
    }
  }

  /**
   * Get available models from OpenRouter API with caching
   * @param params Optional query parameters for filtering
   * @returns Observable of OpenRouter models response
   */
  getAvailableModels(params?: ModelQueryParams): Observable<OpenRouterModelsResponse> {
    // Check cache first
    if (this.isCacheValid()) {
      const cachedModels = this.getCachedModels();
      if (cachedModels) {
        this.allModels = cachedModels;
        return of({ data: cachedModels });
      }
    }
    let url = `${this.openRouterApiUrl}/models`;
    
    // Build query parameters if provided
    if (params) {
      const queryParams = new URLSearchParams();
      
      if (params.category) {
        queryParams.append('category', params.category);
      }
      if (params.use_rss !== undefined) {
        queryParams.append('use_rss', params.use_rss.toString());
      }
      if (params.use_rss_chat_links !== undefined) {
        queryParams.append('use_rss_chat_links', params.use_rss_chat_links.toString());
      }
      
      const queryString = queryParams.toString();
      if (queryString) {
        url += `?${queryString}`;
      }
    }

    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    return this.http.get<OpenRouterModelsResponse>(url, { headers }).pipe(
      tap(response => {
        // Cache the response
        this.cacheModels(response.data);
      })
    );
  }

  /**
   * Get paginated models
   * @param page Page number (1-based)
   * @param pageSize Number of models per page
   * @param searchTerm Optional search term to filter models
   * @returns Observable of paginated models response
   */
  getPaginatedModels(page: number = 1, pageSize: number = 20, searchTerm?: string): Observable<PaginatedModelsResponse> {
    return this.getAvailableModels().pipe(
      map(response => {
        let filteredModels = response.data;
        
        // Apply search filter if provided
        if (searchTerm && searchTerm.trim()) {
          const term = searchTerm.toLowerCase().trim();
          filteredModels = response.data.filter(model => 
            model.name.toLowerCase().includes(term) ||
            model.description.toLowerCase().includes(term) ||
            model.id.toLowerCase().includes(term)
          );
        }
        
        const total = filteredModels.length;
        const totalPages = Math.ceil(total / pageSize);
        const startIndex = (page - 1) * pageSize;
        const endIndex = startIndex + pageSize;
        const paginatedData = filteredModels.slice(startIndex, endIndex);
        
        return {
          data: paginatedData,
          total,
          page,
          pageSize,
          totalPages
        };
      })
    );
  }

  /**
   * Clear models cache
   */
  clearCache(): void {
    localStorage.removeItem(this.cacheKey);
    this.allModels = [];
  }

  /**
   * Get models filtered by category
   * @param category Category to filter by (e.g., 'programming')
   * @returns Observable of filtered models
   */
  getModelsByCategory(category: string): Observable<OpenRouterModelsResponse> {
    return this.getAvailableModels({ category });
  }

  /**
   * Get a specific model by ID
   * @param modelId The model ID to search for
   * @returns Observable of the specific model or null if not found
   */
  getModelById(modelId: string): Observable<OpenRouterModel | null> {
    return new Observable(observer => {
      this.getAvailableModels().subscribe({
        next: (response) => {
          const model = response.data.find(m => m.id === modelId);
          observer.next(model || null);
          observer.complete();
        },
        error: (error) => {
          observer.error(error);
        }
      });
    });
  }

  /**
   * Search models by name or description
   * @param searchTerm Term to search for in model names and descriptions
   * @returns Observable of matching models
   */
  searchModels(searchTerm: string): Observable<OpenRouterModel[]> {
    return new Observable(observer => {
      this.getAvailableModels().subscribe({
        next: (response) => {
          const filteredModels = response.data.filter(model => 
            model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            model.description.toLowerCase().includes(searchTerm.toLowerCase())
          );
          observer.next(filteredModels);
          observer.complete();
        },
        error: (error) => {
          observer.error(error);
        }
      });
    });
  }

  /**
   * Get models that support specific input modalities
   * @param modalities Array of required input modalities (e.g., ['text', 'image'])
   * @returns Observable of models supporting the specified modalities
   */
  getModelsByInputModalities(modalities: string[]): Observable<OpenRouterModel[]> {
    return new Observable(observer => {
      this.getAvailableModels().subscribe({
        next: (response) => {
          const filteredModels = response.data.filter(model => 
            modalities.every(modality => 
              model.architecture.input_modalities.includes(modality)
            )
          );
          observer.next(filteredModels);
          observer.complete();
        },
        error: (error) => {
          observer.error(error);
        }
      });
    });
  }

  /**
   * Get models sorted by pricing (prompt cost)
   * @param ascending Sort in ascending order (cheapest first) if true
   * @returns Observable of models sorted by pricing
   */
  getModelsSortedByPricing(ascending: boolean = true): Observable<OpenRouterModel[]> {
    return new Observable(observer => {
      this.getAvailableModels().subscribe({
        next: (response) => {
          const sortedModels = response.data.sort((a, b) => {
            const priceA = parseFloat(a.pricing.prompt);
            const priceB = parseFloat(b.pricing.prompt);
            return ascending ? priceA - priceB : priceB - priceA;
          });
          observer.next(sortedModels);
          observer.complete();
        },
        error: (error) => {
          observer.error(error);
        }
      });
    });
  }
}
