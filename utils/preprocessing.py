import re
import string

def remove_emojis(text: str) -> str:
    """
    Removes emojis, symbols, and pictographs from the text.
    Uses a comprehensive unicode range regex designed to match emoji blocks.
    
    Parameters:
        text (str): The raw input text.
        
    Returns:
        str: Text with all emojis removed.
    """
    # Regex pattern to match standard emojis, pictographs, dingbats, and technical symbols
    # Unicode ranges:
    # \U00010000-\U0010FFFF matches high unicode code points (emojis, etc.)
    # \u2600-\u27BF matches miscellaneous symbols and dingbats
    # \uD83C-\uD83E][\uDC00-\uDFFF] matches surrogate pairs used for emojis
    emoji_pattern = re.compile(
        r'[\U00010000-\U0010FFFF]|'
        r'[\u2600-\u27BF]|'
        r'[\u2300-\u23FF]|'
        r'[\u2B50-\u2B55]|'
        r'[\u3030]|[\u303D]|[\u3297]|[\u3299]|'
        r'[\uD83C-\uD83E][\uDC00-\uDFFF]'
    )
    return emoji_pattern.sub(r'', text)

def remove_punctuations(text: str) -> str:
    """
    Removes standard punctuation marks as well as Bangla-specific punctuation like the '।'.
    
    Parameters:
        text (str): The input text.
        
    Returns:
        str: Text with all punctuation marks replaced by a single space.
    """
    # Standard English punctuation from Python's string library
    eng_punctuation = string.punctuation
    
    # Bangla-specific punctuation:
    # '।' (Dari - Bangla full-stop)
    # '॥' (Double Dari)
    # '৳' (Bangla Taka sign - optional, but useful to clean up)
    bangla_punctuation = '।' + '॥' + '৳'
    
    all_punctuation = eng_punctuation + bangla_punctuation
    
    # Create a translation table that maps all punctuation to None (removes them)
    # Alternatively, replace punctuation with a space to prevent joining words accidentally
    translator = str.maketrans(all_punctuation, ' ' * len(all_punctuation))
    return text.translate(translator)

def clean_whitespace(text: str) -> str:
    """
    Cleans up extra spaces, tabs, and newlines.
    
    Parameters:
        text (str): The input text.
        
    Returns:
        str: Cleaned text with single spaces and trimmed ends.
    """
    # Replace one or more whitespace characters with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing whitespaces
    return cleaned.strip()

def preprocess_bangla_text(text: str) -> str:
    """
    Full text preprocessing pipeline for Bangla emotional analysis.
    Applies emoji removal, punctuation stripping, and extra whitespace cleanup.
    
    Parameters:
        text (str): The raw input Bangla text (can be a string or list/series).
        
    Returns:
        str: Preprocessed clean text ready for feature extraction (TF-IDF).
    """
    if not isinstance(text, str):
        return ""
        
    # Step 1: Remove Emojis
    text_no_emoji = remove_emojis(text)
    
    # Step 2: Remove Punctuations
    text_no_punct = remove_punctuations(text_no_emoji)
    
    # Step 3: Clean up extra whitespaces
    cleaned_text = clean_whitespace(text_no_punct)
    
    return cleaned_text

# If executed directly, run a quick self-test
if __name__ == "__main__":
    sample = "আজ আমার মন অনেক ভালো! 😊  আমি জিপিএ ৫ পেয়েছি।  "
    print("Original Text: ", repr(sample))
    print("Preprocessed Text: ", repr(preprocess_bangla_text(sample)))
