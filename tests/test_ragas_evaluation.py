import os
import pytest
from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall,
    FactualCorrectness,
    BleuScore,
    ResponseRelevancy,
)

@pytest.mark.integration
def test_ragas_metrics(
    evaluation_dataset,
    ollama_embed_wrapper,
    ollama_llm_wrapper,
):
    os.environ["OPENAI_API_KEY"] = "no-key"

    metrics = [
        LLMContextRecall(),
        FactualCorrectness(),
        BleuScore(),
        ResponseRelevancy(),
    ]

    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        embeddings=ollama_embed_wrapper,
        llm=ollama_llm_wrapper,
    )
    print("RAGAS Evaluation Results:")
    print(results)

    assert results.scores, "scores should not be empty"

    for i, example_score in enumerate(results.scores):
        for metric, value in example_score.items():
            assert isinstance(value, float), f"[{i}] {metric} should be float, got {type(value)}"
            assert 0.5 <= value <= 1.0, f"[{i}] {metric} out of bounds: {value}"
