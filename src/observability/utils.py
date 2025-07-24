import functools
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from loguru import logger
from opik import opik_context
from opik import track as opik_track

TRACING_BACKENDS = {
    "opik": opik_track,
}

env_enabled = os.getenv("TRACING_ENABLED", "false").lower() in ("1", "true", "yes")

def postprocess_opik(result: Dict[str, Any]) -> None:
    if not isinstance(result, dict):
        logger.warning("[Tracing] opik postprocessor: result is not a dict")
        return

    required_keys = [
        "model",
        "eval_duration",
        "load_duration",
        "prompt_eval_duration",
        "prompt_eval_count",
        "eval_count",
        "done",
        "done_reason",
    ]
    for key in required_keys:
        if key not in result:
            logger.warning(f"[Tracing] Key '{key}' missing in response")

    opik_context.update_current_span(
        metadata={
            "model": result.get("model"),
            "eval_duration": result.get("eval_duration"),
            "load_duration": result.get("load_duration"),
            "prompt_eval_duration": result.get("prompt_eval_duration"),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "done": result.get("done"),
            "done_reason": result.get("done_reason"),
        },
        usage={
            "completion_tokens": result.get("eval_count", 0),
            "prompt_tokens": result.get("prompt_eval_count", 0),
            "total_tokens": result.get("eval_count", 0)
                          + result.get("prompt_eval_count", 0),
        },
    )

POSTPROCESSORS: Dict[str, Callable[[Dict[str, Any]], None]] = {
    "opik": postprocess_opik,
}

def trace(tracer: Optional[str] = None, enabled: bool = True, **trace_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Generic decorator for dynamic tracing with optional post-processing hooks.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not enabled or not env_enabled:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return wrapper

        backend_decorator = TRACING_BACKENDS.get(tracer or "")
        postprocessor = POSTPROCESSORS.get(tracer or "")

        if backend_decorator:
            decorated_func = backend_decorator(**trace_kwargs)(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = decorated_func(*args, **kwargs)
                if postprocessor:
                    try:
                        postprocessor(result)
                    except Exception as e:
                        logger.warning(f"[Tracing] Postprocessing failed: {e}")
                return result
            return wrapper
        else:
            logger.debug(f"[Tracing] No valid tracer found for '{tracer}'")
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return wrapper

    return decorator

def setup_tracing() -> Any:
    """
    Dynamically enables or disables tracing based on the TRACING_ENABLED environment variable.
    If disabled, returns no-op equivalents for @track and opik_context.update_current_span.

    Returns:
        An object with `.track` and `.opik_context.update_current_span(...)` available.
    """

    if not env_enabled:
        logger.info("[Tracing] Tracing disabled via TRACING_ENABLED")
        return SimpleNamespace(
            track=lambda *args, **kwargs: (lambda f: f),
            opik_context=SimpleNamespace(
                update_current_span=lambda *args, **kwargs: None
            ),
        )

    try:
        import opik
        opik.configure(use_local=True, automatic_approvals=True)
        logger.info("[Tracing] Opik tracing enabled and configured")
        return opik
    except Exception as e:
        logger.warning(f"[Tracing] Failed to configure Opik: {e}")
        return SimpleNamespace(
            track=lambda *args, **kwargs: (lambda f: f),
            opik_context=SimpleNamespace(
                update_current_span=lambda *args, **kwargs: None
            ),
        )
