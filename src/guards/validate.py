import re
from langdetect import detect_langs
from guards import get_guards
from loguru import logger
from unidecode import unidecode

GUARDS_FAILED_MSG = "Your input is not appropriate pls try again."

def is_english(text: str) -> bool:
    langs = detect_langs(text)
    return langs[0].lang == "en" and langs[0].prob > 0.85

def is_garbage(text: str) -> bool:
    if len(text) < 3:
        return True
    if re.fullmatch(r'[\W_]+', text):
        return True
    if re.search(r'(.)\1{10,}', text):
        return True
    return False

def sanitize_input(text: str) -> str:
    cleaned = unidecode(text)
    return cleaned

def validate_input(text:str) -> str:
    if is_garbage(text) or not is_english(text):
        raise ValueError(GUARDS_FAILED_MSG)
    
    try:
        validation_outcome = get_guards().validate(text)
        logger.debug(validation_outcome.validation_summaries)
    except Exception as e:
        raise ValueError(GUARDS_FAILED_MSG)

    return sanitize_input(str)
