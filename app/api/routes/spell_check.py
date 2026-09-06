from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from app.services.spell_corrector import get_spell_corrector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spell-check", tags=["Spell Check"])

class SpellCheckRequest(BaseModel):
    text: str
    max_distance: Optional[int] = 2

class WordCorrectionRequest(BaseModel):
    word: str
    max_distance: Optional[int] = 2

class SuggestionsRequest(BaseModel):
    word: str
    max_distance: Optional[int] = 2
    max_results: Optional[int] = 5

class SpellCheckResponse(BaseModel):
    original_text: str
    corrected_text: str
    corrections_made: bool
    processing_time_ms: float

class WordCorrectionResponse(BaseModel):
    original_word: str
    corrected_word: str
    was_corrected: bool
    processing_time_ms: float

class SuggestionsResponse(BaseModel):
    word: str
    suggestions: List[str]
    processing_time_ms: float

class CacheStatsResponse(BaseModel):
    hits: int
    misses: int
    total_requests: int
    cache_size: int
    hit_rate: float

@router.post("/correct-text", response_model=SpellCheckResponse)
async def correct_text(request: SpellCheckRequest):
    """
    Correct spelling errors in a text string.
    
    This endpoint corrects spelling errors in both Persian and English text
    using Levenshtein distance algorithm with dictionary lookup.
    
    - **text**: The text to be spell-checked and corrected
    - **max_distance**: Maximum Levenshtein distance for corrections (default: 2)
    """
    try:
        import time
        start_time = time.time()
        
        corrector = get_spell_corrector()
        corrected_text = corrector.correct_text(request.text, request.max_distance)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SpellCheckResponse(
            original_text=request.text,
            corrected_text=corrected_text,
            corrections_made=request.text != corrected_text,
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        logger.error(f"Error in spell check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Spell check failed: {str(e)}")

@router.post("/correct-word", response_model=WordCorrectionResponse)
async def correct_word(request: WordCorrectionRequest):
    """
    Correct spelling of a single word.
    
    This endpoint corrects a single word using the spell corrector's dictionary
    and Levenshtein distance algorithm.
    
    - **word**: The word to be corrected
    - **max_distance**: Maximum Levenshtein distance for corrections (default: 2)
    """
    try:
        import time
        start_time = time.time()
        
        corrector = get_spell_corrector()
        corrected_word = corrector.correct_word(request.word, request.max_distance)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return WordCorrectionResponse(
            original_word=request.word,
            corrected_word=corrected_word,
            was_corrected=request.word != corrected_word,
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        logger.error(f"Error in word correction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Word correction failed: {str(e)}")

@router.post("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(request: SuggestionsRequest):
    """
    Get multiple spelling correction suggestions for a word.
    
    This endpoint returns multiple possible corrections for a misspelled word,
    ranked by Levenshtein distance.
    
    - **word**: The word to get suggestions for
    - **max_distance**: Maximum Levenshtein distance to consider (default: 2)
    - **max_results**: Maximum number of suggestions to return (default: 5)
    """
    try:
        import time
        start_time = time.time()
        
        corrector = get_spell_corrector()
        suggestions = corrector.get_correction_suggestions(
            request.word, 
            request.max_distance, 
            request.max_results
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SuggestionsResponse(
            word=request.word,
            suggestions=suggestions,
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Getting suggestions failed: {str(e)}")

@router.get("/cache-stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get spell corrector cache performance statistics.
    
    This endpoint returns detailed statistics about the spell corrector's
    cache performance including hit rate, cache size, and total requests.
    """
    try:
        corrector = get_spell_corrector()
        stats = corrector.get_cache_stats()
        
        return CacheStatsResponse(
            hits=stats["hits"],
            misses=stats["misses"],
            total_requests=stats["total_requests"],
            cache_size=stats["cache_size"],
            hit_rate=round(stats["hit_rate"] * 100, 2)  # Convert to percentage
        )
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Getting cache stats failed: {str(e)}")

@router.post("/clear-cache")
async def clear_cache():
    """
    Clear the spell corrector cache.
    
    This endpoint clears all cached corrections and resets cache statistics.
    Useful for testing or when you want to start fresh.
    """
    try:
        corrector = get_spell_corrector()
        corrector.clear_cache()
        
        return {"message": "Cache cleared successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clearing cache failed: {str(e)}")

@router.get("/dictionary-info")
async def get_dictionary_info():
    """
    Get information about the loaded dictionary.
    
    This endpoint returns information about the spell corrector's dictionary
    including the number of words loaded and dictionary path.
    """
    try:
        corrector = get_spell_corrector()
        
        return {
            "dictionary_path": corrector.dictionary_path,
            "total_words": len(corrector.dictionary),
            "cache_max_size": corrector.max_cache_size,
            "status": "loaded"
        }
    except Exception as e:
        logger.error(f"Error getting dictionary info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Getting dictionary info failed: {str(e)}")