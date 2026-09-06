import re
import json
import os
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache
from pathlib import Path
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

class SpellCorrector:
    """Spell correction service using Levenshtein Distance algorithm.
    
    This class provides spell correction functionality for both Persian and English text
    using a dictionary-based approach with Levenshtein Distance for finding the closest matches.
    Includes caching for improved performance.
    """
    
    def __init__(self, dictionary_path: Optional[str] = None, cache_size: int = 1000):
        """Initialize the spell corrector.
        
        Args:
            dictionary_path: Path to the dictionary file. If None, uses default path.
            cache_size: Maximum number of cached corrections
        """
        self.dictionary_path = dictionary_path or self._get_default_dictionary_path()
        self.dictionary: Set[str] = set()
        self.cache: Dict[str, str] = {}
        self.max_cache_size = cache_size
        
        # Initialize cache with OrderedDict for LRU behavior
        self._correction_cache = OrderedDict()
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
        
        self._load_dictionary()
    
    def _get_default_dictionary_path(self) -> str:
        """Get the default dictionary file path."""
        current_dir = Path(__file__).parent
        return str(current_dir / "dictionaries" / "words.json")
    
    def _load_dictionary(self):
        """Load dictionary from file."""
        try:
            if os.path.exists(self.dictionary_path):
                with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Support both list format and dict format
                    if isinstance(data, list):
                        self.dictionary = set(word.lower() for word in data)
                    elif isinstance(data, dict):
                        words = []
                        for lang_words in data.values():
                            if isinstance(lang_words, list):
                                words.extend(lang_words)
                        self.dictionary = set(word.lower() for word in words)
                    logger.info(f"Loaded {len(self.dictionary)} words from dictionary")
            else:
                logger.warning(f"Dictionary file not found at {self.dictionary_path}. Using empty dictionary.")
                self.dictionary = set()
        except Exception as e:
            logger.error(f"Error loading dictionary: {str(e)}")
            self.dictionary = set()
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            The Levenshtein distance between the strings
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def find_closest_matches(self, word: str, max_distance: int = 2, max_results: int = 5) -> List[Tuple[str, int]]:
        """Find closest matches for a word in the dictionary.
        
        Args:
            word: The word to find matches for
            max_distance: Maximum Levenshtein distance to consider
            max_results: Maximum number of results to return
            
        Returns:
            List of tuples (word, distance) sorted by distance
        """
        word_lower = word.lower()
        matches = []
        
        for dict_word in self.dictionary:
            distance = self.levenshtein_distance(word_lower, dict_word)
            if distance <= max_distance:
                matches.append((dict_word, distance))
        
        # Sort by distance and return top results
        matches.sort(key=lambda x: x[1])
        return matches[:max_results]
    
    def correct_word(self, word: str, max_distance: int = 2) -> str:
        """Correct a single word using Levenshtein distance with caching.
        
        Args:
            word: The word to correct
            max_distance: Maximum Levenshtein distance to consider
            
        Returns:
            The corrected word or the original word if no correction found
        """
        if not word or not word.strip():
            return word
            
        word_lower = word.lower()
        self._cache_stats["total_requests"] += 1
        
        # Check cache first
        if word_lower in self._correction_cache:
            self._cache_stats["hits"] += 1
            # Move to end (most recently used)
            self._correction_cache.move_to_end(word_lower)
            return self._correction_cache[word_lower]
            
        self._cache_stats["misses"] += 1
        
        # If word is already in dictionary, return as is
        if word_lower in self.dictionary:
            result = word
        else:
            # Find closest matches
            matches = self.find_closest_matches(word, max_distance)
            if matches:
                # Return the closest match (first in sorted list)
                closest_word = matches[0][0]
                # Preserve original case pattern
                result = self._preserve_case(word, closest_word)
            else:
                result = word
        
        # Cache the result
        self._update_cache(word_lower, result)
        
        return result
    
    def _preserve_case(self, original: str, corrected: str) -> str:
        """Preserve the case pattern of the original word in the corrected word.
        
        Args:
            original: The original word with its case pattern
            corrected: The corrected word in lowercase
            
        Returns:
            The corrected word with the original case pattern applied
        """
        if original.isupper():
            return corrected.upper()
        elif original.istitle():
            return corrected.capitalize()
        elif original.islower():
            return corrected.lower()
        else:
            # Mixed case - apply character by character where possible
            result = []
            for i, char in enumerate(corrected):
                if i < len(original):
                    if original[i].isupper():
                        result.append(char.upper())
                    else:
                        result.append(char.lower())
                else:
                    result.append(char.lower())
            return ''.join(result)
    
    def correct_text(self, text: str, max_distance: int = 2) -> str:
        """Correct spelling in a text string.
        
        Args:
            text: The text to correct
            max_distance: Maximum Levenshtein distance to consider for corrections
            
        Returns:
            The text with spelling corrections applied
        """
        if not text or not text.strip():
            return text
        
        # Split text into words while preserving punctuation and whitespace
        # This regex keeps words, punctuation, and whitespace separate
        tokens = re.findall(r'\S+|\s+', text)
        
        corrected_tokens = []
        for token in tokens:
            if token.isspace():
                # Preserve whitespace as is
                corrected_tokens.append(token)
            else:
                # Extract words from token (removing punctuation)
                words = re.findall(r'\b\w+\b', token)
                if words:
                    # Get the main word (usually there's only one)
                    word = words[0]
                    corrected_word = self.correct_word(word, max_distance)
                    # Replace the word in the original token
                    corrected_token = token.replace(word, corrected_word, 1)
                    corrected_tokens.append(corrected_token)
                else:
                    # No words found (pure punctuation), keep as is
                    corrected_tokens.append(token)
        
        return ''.join(corrected_tokens)
    
    def add_words_to_dictionary(self, words: List[str]):
        """Add words to the dictionary.
        
        Args:
            words: List of words to add
        """
        for word in words:
            self.dictionary.add(word.lower())
        logger.info(f"Added {len(words)} words to dictionary")
    
    def save_dictionary(self):
        """Save the current dictionary to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.dictionary_path), exist_ok=True)
            
            # Save as a simple list
            with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(sorted(list(self.dictionary)), f, ensure_ascii=False, indent=2)
            logger.info(f"Dictionary saved to {self.dictionary_path}")
        except Exception as e:
            logger.error(f"Error saving dictionary: {str(e)}")
    
    def get_correction_suggestions(self, word: str, max_distance: int = 2, max_results: int = 5) -> List[str]:
        """Get multiple correction suggestions for a word.
        
        Args:
            word: The word to get suggestions for
            max_distance: Maximum Levenshtein distance to consider
            max_results: Maximum number of suggestions to return
            
        Returns:
            List of suggested corrections
        """
        matches = self.find_closest_matches(word, max_distance, max_results)
        return [self._preserve_case(word, match[0]) for match, _ in matches]
    
    def _update_cache(self, word: str, correction: str):
        """Update the correction cache with LRU behavior.
        
        Args:
            word: The original word (key)
            correction: The corrected word (value)
        """
        if word in self._correction_cache:
            # Update existing entry and move to end
            self._correction_cache[word] = correction
            self._correction_cache.move_to_end(word)
        else:
            # Add new entry
            if len(self._correction_cache) >= self.max_cache_size:
                # Remove least recently used item
                self._correction_cache.popitem(last=False)
            self._correction_cache[word] = correction
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary containing cache hit/miss statistics
        """
        stats = self._cache_stats.copy()
        stats["cache_size"] = len(self._correction_cache)
        if stats["total_requests"] > 0:
            stats["hit_rate"] = stats["hits"] / stats["total_requests"]
        else:
            stats["hit_rate"] = 0.0
        return stats
    
    def clear_cache(self):
        """Clear the correction cache."""
        self.cache.clear()
        self._correction_cache.clear()
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
        logger.info("Spell correction cache cleared")


# Singleton instance
_spell_corrector = None

def get_spell_corrector() -> SpellCorrector:
    """Get the spell corrector instance.
    
    Returns:
        A singleton instance of the SpellCorrector
    """
    global _spell_corrector
    if _spell_corrector is None:
        _spell_corrector = SpellCorrector()
    return _spell_corrector