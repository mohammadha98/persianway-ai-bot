export type SearchDepth = 'basic' | 'advanced';

export interface TavilySearchSettings {
  tavily_api_key: string | null;
  is_enabled: boolean;
  search_depth: SearchDepth;
  max_results: number;
  include_answer: boolean;
  include_domains: string[];
  exclude_domains: string[];
  snippet_length: number;
}

export interface ApiResponse<T> {
  success: boolean;
  message: string;
  settings: T;
}
