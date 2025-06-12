from guardrails.hub import ToxicLanguage, DetectJailbreak
from guardrails import Guard


def get_guards() -> Guard:
    return Guard().use_many(
        DetectJailbreak,
        ToxicLanguage(
            use_local=True,
            threshold=0.5,
            validation_method="sentence",
            on_fail="exception",
        ),
    )
