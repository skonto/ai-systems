from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import OllamaLLM
from ragas.metrics import LLMContextRecall, BleuScore, FactualCorrectness

import os
from rag import OllamaRag
from rag.prompts import get_initial_chat_state
from langchain_ollama import OllamaEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# Original QA pairs (shortened for brevity — use full set for production)
qa_data = {
    "What is FantasticCharge Pro?": "FantasticCharge Pro is a compact, portable power bank designed to charge multiple devices quickly and efficiently.",
    "Who is FantasticCharge Pro designed for?": "It is perfect for travelers, remote workers, gamers, and anyone needing reliable on-the-go charging.",
    "How does FantasticCharge Pro work?": "FantasticCharge Pro stores energy in its high-capacity lithium-ion battery, delivering fast and efficient charging to your devices via USB-C and USB-A ports.",
    "Is FantasticCharge Pro safe to use?": "Yes, it features advanced safety mechanisms such as overcharge protection, short circuit prevention, and temperature control.",
}

# Manually reworded questions for testing
rephrased_questions = {
    "What is FantasticCharge Pro?": "Can you describe what FantasticCharge Pro is?",
    "Who is FantasticCharge Pro designed for?": "What types of users is FantasticCharge Pro ideal for?",
    "How does FantasticCharge Pro work?": "What is the mechanism behind how FantasticCharge Pro functions?",
    "Is FantasticCharge Pro safe to use?": "Are there any safety features in FantasticCharge Pro?",
}


def main():
    rag_pipeline = OllamaRag()
    dataset = []

    for orig_q, new_q in rephrased_questions.items():
        ground_truth = qa_data[orig_q]
        initial_state = get_initial_chat_state()

        # Get model response and retrieved contexts
        answer, contexts = rag_pipeline.get_response(new_q, initial_state)

        dataset.append({
            "user_input": new_q,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": ground_truth,
        })

    os.environ['OPENAI_API_KEY'] = 'no-key'

    ollama_e = OllamaEmbeddings(model="bge-m3",base_url="http://localhost:11434")
    ollama_embed_self = LangchainEmbeddingsWrapper(ollama_e)
    ollama_llm = OllamaLLM(model="llama3:8b-instruct-q4_0",
                           temperature=0, top_k=1, seed=1234, num_predict=1000,)
    wrapper = LangchainLLMWrapper(ollama_llm)

    metrics=  [LLMContextRecall(),  FactualCorrectness(), BleuScore()]
    from ragas import EvaluationDataset
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    # Evaluate with RAGAS
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics, 
        embeddings=ollama_embed_self,
        llm=wrapper
    )

    print("✅ RAGAS Evaluation Results:")
    print(results)


if __name__ == "__main__":
    main()
