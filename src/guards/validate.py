import re

from langdetect import detect_langs
from loguru import logger
from unidecode import unidecode

from guards import get_guards

GUARDS_FAILED_MSG = "Your input is not appropriate pls try again."


def is_english(text: str, threshold: float = 0.85) -> bool:
    """
    Determines whether the given text is in English based on language detection confidence.

    Uses `langdetect.detect_langs` to determine the most probable language and its confidence score.
    Returns True only if English is the top prediction and its probability exceeds the threshold.

    Args:
        text (str): The input text to analyze.
        threshold (float, optional): Minimum probability required to consider the text English.
                                     Defaults to 0.85.

    Returns:
        bool: True if the text is confidently detected as English, False otherwise.
    """
    langs = detect_langs(text)
    return langs[0].lang == "en" and langs[0].prob >= threshold


def is_garbage(text: str) -> bool:
    """
    Determines whether the input text is considered garbage.

    A text is classified as garbage if it meets any of the following:
    - It consists entirely of non-alphanumeric characters (symbols, punctuation, etc.).
    - It contains excessive character repetition (e.g., "aaaaaaaaaaaa").

    Args:
        text (str): The input text to evaluate.

    Returns:
        bool: True if the text is classified as garbage, False otherwise.
    """
    if re.fullmatch(r"[\W_]+", text):
        return True
    if re.search(r"(.)\1{10,}", text):
        return True
    return False


def sanitize_input(text: str) -> str:
    """
    Convert Unicode input text to closest ASCII representation.

    Args:
        text (str): Input text to sanitize.

    Returns:
        str: ASCII-only sanitized text.
    """
    return unidecode(text)


def validate_input(text: str) -> str:
    """
    Validates and sanitizes input text.

    The input is first checked for garbage content and language. If it passes,
    it is further validated using external guard rules. Any validation failure
    raises a ValueError. On success, the text is sanitized and returned.

    Args:
        text (str): The user-provided input to validate.

    Returns:
        str: Sanitized and validated ASCII-only text.

    Raises:
        ValueError: If the input is considered garbage, non-English, or fails
                    external guard validation.
    """
    if is_garbage(text) or not is_english(text):
        print(is_garbage(text))
        print(is_english(text))
        raise ValueError("Your input text is not appropriate")

    try:
        validation_outcome = get_guards().validate(text)
        logger.debug(validation_outcome.validation_summaries)
    except Exception as exc:
        print(exc)
        raise ValueError(GUARDS_FAILED_MSG) from exc

    return sanitize_input(text)
