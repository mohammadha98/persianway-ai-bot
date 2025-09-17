#!/usr/bin/env python3
"""
Test script for the spell corrector functionality.
This script tests the spell correction with various Persian and English words.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.spell_corrector import get_spell_corrector

def test_spell_correction():
    """Test the spell correction functionality with sample inputs."""
    print("Testing Spell Corrector...")
    print("=" * 50)
    
    # Get the spell corrector instance
    corrector = get_spell_corrector()
    
    # Test cases with intentional misspellings
    test_cases = [
        # Persian words with typos
        ("سلم", "سلام"),  # Missing 'ا'
        ("خداحافظ", "خداحافظ"),  # Correct word
        ("ممنن", "ممنون"),  # Missing 'و'
        ("خانه", "خانه"),  # Correct word
        ("کتب", "کتاب"),  # Missing 'ا'
        
        # English words with typos
        ("helo", "hello"),  # Missing 'l'
        ("thnks", "thanks"),  # Missing 'a'
        ("computer", "computer"),  # Correct word
        ("frend", "friend"),  # Missing 'i'
        ("scool", "school"),  # Missing 'h'
        
        # Mixed text
        ("سلم hello", "سلام hello"),
        ("helo سلام", "hello سلام"),
    ]
    
    print("Individual Word Corrections:")
    print("-" * 30)
    
    for original, expected in test_cases:
        corrected = corrector.correct_text(original)
        status = "✓" if corrected == expected else "✗"
        print(f"{status} '{original}' → '{corrected}' (expected: '{expected}')")
    
    print("\nCache Statistics:")
    print("-" * 20)
    stats = corrector.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nTesting sentence correction:")
    print("-" * 30)
    
    sentences = [
        "سلم دوست من چطوری؟",
        "helo my frend how are you?",
        "ممنن از کمکتون",
        "thnks for your hlp"
    ]
    
    for sentence in sentences:
        corrected = corrector.correct_text(sentence)
        print(f"Original:  {sentence}")
        print(f"Corrected: {corrected}")
        print()

if __name__ == "__main__":
    test_spell_correction()