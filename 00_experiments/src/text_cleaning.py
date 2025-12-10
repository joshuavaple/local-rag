import re


def clean_text(text: str) -> str:
    """
    Clean text by removing leading/trailing spaces and fixing common spacing issues.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with proper spacing
    """
    if not text:
        return text
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Remove extra spaces between words
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spaces around punctuation
    # Remove spaces before commas, periods, semicolons, colons, exclamation marks, question marks
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    
    # Remove spaces after opening brackets/parentheses
    text = re.sub(r'([(\[\{])\s+', r'\1', text)
    
    # Remove spaces before closing brackets/parentheses
    text = re.sub(r'\s+([)\]\}])', r'\1', text)
    
    # Add single space after punctuation if not already present (except at end of string)
    text = re.sub(r'([,.;:!?])(?=[^\s])', r'\1 ', text)
    
    # Fix spaces around quotes
    text = re.sub(r'\s*(["\'])\s*', r'\1', text)
    text = re.sub(r'(["\'])\s+', r'\1 ', text)
    text = re.sub(r'\s+(["\'])', r' \1', text)
    
    # Remove spaces around hyphens in compound words
    text = re.sub(r'\s*-\s*', '-', text)
    
    # Final cleanup of any remaining multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()