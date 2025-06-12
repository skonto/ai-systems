from guardrails import Guard
from guardrails.hub import DetectJailbreak, ToxicLanguage


def get_guards() -> Guard:
    """
    Initializes and returns a Guard object with multiple safety filters enabled.

    Returns:
        Guard: A configured Guard instance with the following checks:
            - DetectJailbreak: Detects attempts to bypass LLM constraints or safety boundaries.
            - ToxicLanguage: Filters toxic or unsafe language using a local model.
                - Threshold: 0.5 (moderate sensitivity)
                - Validation method: Sentence-level analysis
                - Action on failure: Raises an exception

    This setup is intended for securing LLM applications against prompt injections and harmful outputs.
    """
    return Guard().use_many(
        DetectJailbreak,
        ToxicLanguage(
            use_local=True,
            threshold=0.5,
            validation_method="sentence",
            on_fail="exception",
        ),
    )
