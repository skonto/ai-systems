import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import ollama
from langchain_chroma import Chroma
from langchain_core import documents
from langchain_ollama import OllamaEmbeddings
from loguru import logger
from opik import opik_context, track

from rag.prompts import format_prompt


class OllamaRag:
    """
    A configurable Retrieval-Augmented Generation (RAG) pipeline using:

    - Ollama for LLM-based chat completion and embeddings.
    - Chroma as the persistent vector store.

    This class supports document ingestion, retrieval with relevance scoring,
    prompt formatting, and generating contextual responses from a local LLM.
    """

    def __init__(
        self,
        collection_name: str = "qas",
        db_path: str = "/tmp/ch_db",
        ollama_host: Optional[str] = None,
        model_name: str = "llama3:8b-instruct-q4_0",
        embedding_model: str = "BGE-M3",
        num_gpu: int = 0,
        keep_alive: int = -1,
        temperature: float = 0.0,
        seed: int = 1234,
        top_k: int = 1,
        num_predict: int = 400,
        score_threshold: float = 0.4,
    ):
        """
        Initializes the RAG system with configurable parameters.

        Args:
            collection_name (str): Name of the Chroma vector store collection.
            db_path (str): Path to store the vector DB on disk.
            ollama_host (Optional[str]): Base URL for the Ollama server.
            model_name (str): LLM model identifier used for chat generation.
            embedding_model (str): Embedding model name for document encoding.
            num_gpu (int): Number of GPUs to use for embeddings (0 = CPU).
            keep_alive (int): Ollama session timeout in seconds (-1 for infinite).
            temperature (float): Sampling temperature for LLM generation.
            seed (int): Random seed for deterministic generation.
            top_k (int): Top-k token sampling for decoding.
            num_predict (int): Max number of tokens to generate.
            score_threshold (float): Minimum similarity score to include a document.
        """
        self.collection_name = collection_name
        self.db_path = db_path
        self.model_base_url = ollama_host or os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.top_k = top_k
        self.num_predict = num_predict
        self.score_threshold = score_threshold

        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=self.model_base_url,
            num_gpu=num_gpu,
            keep_alive=keep_alive,
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )

    def get_response(self, text: str, msgs: List[Dict[str, str]]) -> Tuple[str, List[str]]:
        """
        Generates a LLM response based on user input and retrieved document context.

        Args:
            text (str): The user's question or prompt.
            msgs (List[Dict[str, str]]): Message history (OpenAI-style chat format).

        Returns:
            str: The content of the model's generated response.
        """
        chunks = []
        if self.is_collection_empty():
            logger.warning("Collection is empty!")
            retrieved_context = ""
        else:
            results = self.vector_store.similarity_search_with_relevance_scores(
                text, k=5, score_threshold=self.score_threshold
            )

            retrieved_context = ""
            for doc, score in results:
                logger.debug(f"Chunk Score: {score}")
                logger.debug(f"Chunk: {doc.page_content}\n-------\n")
                retrieved_context += doc.page_content
                chunks.append(doc.page_content)

        prompt = format_prompt(question=text, context=retrieved_context)
        conversation = msgs + [{"role": "user", "content": prompt}]

        logger.debug("Chunks passed to LLM:")
        for msg in conversation:
            logger.debug(msg)

        output = self.ollama_llm_call(conversation)
        return (output["message"]["content"], chunks)

    def is_collection_empty(self) -> bool:
        """
        Checks if the vector store has any stored document chunks.

        Returns:
            bool: True if the collection is empty, False otherwise.
        """
        ids = self.vector_store.get().get("ids", [])
        return len(ids) == 0

    def ingest_docs(self, file_path: str, db_path: str) -> None:
        """
        Ingests a structured plain text file into the vector store.

        The file is split on "---" as chunk separators, and each chunk is stored
        as a separate document in the Chroma collection.

        Args:
            file_path (str): Path to the input `.txt` file to ingest.
            db_path(str): Path to the Chroma vector database.
        """
        if self.is_collection_empty():

            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=db_path,
            )

            logger.info(f"Ingesting documents from {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            chunks = [chunk for chunk in content.split("---") if chunk.strip()]
            docs = [documents.Document(page_content=c) for c in chunks]
            ids = [str(uuid4()) for _ in docs]

            self.vector_store.add_documents(documents=docs, ids=ids)
        else:
            logger.info("Skipping ingestion — collection already populated.")

    @track(tags=["ollama", "python-library"])
    def ollama_llm_call(self, msgs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Sends a chat prompt to the Ollama model and logs performance + usage metadata.

        Args:
            msgs (List[Dict[str, str]]): List of messages to send to the LLM
                (in OpenAI-compatible chat format).

        Returns:
            Dict[str, Any]: The complete response from the Ollama LLM,
                including model output, token usage, and timing metadata.
        """
        logger.debug("Sending messages to Ollama:")
        for msg in msgs:
            logger.debug(msg)

        response = ollama.chat(
            model=self.model_name,
            keep_alive=-1,
            messages=msgs,
            options={
                "temperature": self.temperature,
                "seed": self.seed,
                "top_k": self.top_k,
                "num_predict": self.num_predict,
            },
        )

        required_keys = [
            "model", "eval_duration", "load_duration",
            "prompt_eval_duration", "prompt_eval_count",
            "eval_count", "done", "done_reason"
        ]
        for key in required_keys:
            if key not in response:
                logger.warning(f"Key '{key}' missing in Ollama response")

        opik_context.update_current_span(
            metadata={
                "model": response.get("model"),
                "eval_duration": response.get("eval_duration"),
                "load_duration": response.get("load_duration"),
                "prompt_eval_duration": response.get("prompt_eval_duration"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "done": response.get("done"),
                "done_reason": response.get("done_reason"),
            },
            usage={
                "completion_tokens": response.get("eval_count", 0),
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "total_tokens": response.get("eval_count", 0) + \
                      response.get("prompt_eval_count", 0),
            },
        )

        return response
