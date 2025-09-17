from app.services.spell_corrector import SpellCorrector
import time

def test_cache_performance():
    print("Testing Spell Corrector Cache Performance...")
    print("=" * 50)
    
    corrector = SpellCorrector()
    
    # Test words that should be corrected
    test_words = [
        "سلم",  # Should correct to سلام
        "ممنن",  # Should correct to ممنون  
        "helo",  # Should correct to hello
        "thnks", # Should correct to thanks
        "frend", # Should correct to friend
    ]
    
    print("Initial cache statistics:")
    stats = corrector.get_cache_stats()
    print(f"Hits: {stats['hits']}, Misses: {stats['misses']}, Hit Rate: {stats['hit_rate']:.2f}%")
    
    print("\nFirst round - populating cache:")
    start_time = time.time()
    for word in test_words:
        corrected = corrector.correct_word(word)
        print(f"'{word}' -> '{corrected}'")
    first_round_time = time.time() - start_time
    
    print(f"\nFirst round took: {first_round_time:.4f} seconds")
    stats = corrector.get_cache_stats()
    print(f"Cache stats: Hits: {stats['hits']}, Misses: {stats['misses']}, Hit Rate: {stats['hit_rate']:.2f}%")
    
    print("\nSecond round - using cache:")
    start_time = time.time()
    for word in test_words:
        corrected = corrector.correct_word(word)
        print(f"'{word}' -> '{corrected}'")
    second_round_time = time.time() - start_time
    
    print(f"\nSecond round took: {second_round_time:.4f} seconds")
    stats = corrector.get_cache_stats()
    print(f"Cache stats: Hits: {stats['hits']}, Misses: {stats['misses']}, Hit Rate: {stats['hit_rate']:.2f}%")
    
    # Performance improvement calculation
    if first_round_time > 0:
        improvement = ((first_round_time - second_round_time) / first_round_time) * 100
        print(f"\nPerformance improvement: {improvement:.1f}%")
    
    print("\nTesting text correction with cache:")
    sentences = [
        "سلم دوست من چطوری؟",
        "helo my frend how are you?",
        "ممنن از کمکتون",
    ]
    
    for sentence in sentences:
        start_time = time.time()
        corrected = corrector.correct_text(sentence)
        correction_time = time.time() - start_time
        print(f"Original: {sentence}")
        print(f"Corrected: {corrected}")
        print(f"Time: {correction_time:.4f} seconds")
        print()
    
    print("Final cache statistics:")
    stats = corrector.get_cache_stats()
    print(f"Hits: {stats['hits']}, Misses: {stats['misses']}, Hit Rate: {stats['hit_rate']:.2f}%")
    print(f"Total corrections cached: {len(corrector._correction_cache)}")

if __name__ == "__main__":
    test_cache_performance()