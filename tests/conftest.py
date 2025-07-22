import os
import pytest

from rag import OllamaRag
from rag.prompts import get_initial_chat_state

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset


qa_data = {
    "What is FantasticCharge Pro?": "FantasticCharge Pro is a compact, portable power bank designed to charge multiple devices quickly and efficiently.",
    "Who is FantasticCharge Pro designed for?": "It is perfect for travelers, remote workers, gamers, and anyone needing reliable on-the-go charging.",
    "How does FantasticCharge Pro work?": "FantasticCharge Pro stores energy in its high-capacity lithium-ion battery, delivering fast and efficient charging to your devices via USB-C and USB-A ports.",
    "Is FantasticCharge Pro safe to use?": "Yes, it features advanced safety mechanisms such as overcharge protection, short circuit prevention, and temperature control.",
}

rephrased_questions = {
    "What is FantasticCharge Pro?": "Can you describe what FantasticCharge Pro is?",
    "Who is FantasticCharge Pro designed for?": "What types of users is FantasticCharge Pro ideal for?",
    "How does FantasticCharge Pro work?": "What is the mechanism behind how FantasticCharge Pro functions?",
    "Is FantasticCharge Pro safe to use?": "Are there any safety features in FantasticCharge Pro?",
}


@pytest.fixture(scope="module")
def rag_pipeline():
    return OllamaRag()


@pytest.fixture(scope="module")
def evaluation_dataset(rag_pipeline):
    dataset = []

    for orig_q, new_q in rephrased_questions.items():
        ground_truth = qa_data[orig_q]
        initial_state = get_initial_chat_state()

        answer, contexts = rag_pipeline.get_response(new_q, initial_state)

        dataset.append(
            {
                "user_input": new_q,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": ground_truth,
            }
        )

    return EvaluationDataset.from_list(dataset)


@pytest.fixture(scope="module")
def ollama_llm_wrapper():
    llm = OllamaLLM(
        model="llama3:8b-instruct-q4_0",
        temperature=0,
        top_k=1,
        seed=1234,
        num_predict=1000,
    )
    return LangchainLLMWrapper(llm)


@pytest.fixture(scope="module")
def ollama_embed_wrapper():
    emb = OllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")
    return LangchainEmbeddingsWrapper(emb)
