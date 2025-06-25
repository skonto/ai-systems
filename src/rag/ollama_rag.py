import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from chromadb.config import Settings
import chromadb
import ollama
from langchain_chroma import Chroma
from langchain_core import documents
from langchain_ollama import OllamaEmbeddings
from loguru import logger
from opik import opik_context, track

from rag.prompts import format_prompt
from rag.utils import build_langchain_bm25_retriever, fuse_with_bm25


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
        num_predict: int = 1000,
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

        # TODO: move to Milvus, elastic etc this is only for local
        self.bm25 = build_langchain_bm25_retriever(self.get_all_documents_from_collection(self.collection_name, self.db_path), 5)

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
        new_text = text 
        if self.needs_rewrite(text):
            new_text = self.rewrite_ambiguous_prompt(msgs, text)

        if self.is_collection_empty():
            logger.warning("Collection is empty!")
            retrieved_context = ""
        else:
            print(text)
            results = self.vector_store.similarity_search_with_relevance_scores(
                new_text, k=5, score_threshold=self.score_threshold
            )

            retrieved_context = ""
            for doc, score in results:
                logger.debug(f"Chunk Score: {score}")
                logger.debug(f"Chunk: {doc.page_content}\n-------\n")
                chunks.append(doc.page_content)
            fused_results = fuse_with_bm25(results, self.bm25, new_text, alpha=0.4)
            retrieved_context = "".join(doc.page_content + "\n" for doc, _ in fused_results)

        prompt = format_prompt(question=new_text, context=retrieved_context)

        conversation = msgs + [{"role": "user", "content": prompt}]

        logger.debug("Chunks passed to LLM:")
        for msg in conversation:
            logger.debug(msg)

        output = self.ollama_llm_call(conversation)
        return (output["message"]["content"].strip(), chunks)

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
            path = Path(file_path)
            files = []

            if path.is_file() and path.suffix == ".txt":
                files = [path]
            elif path.is_dir():
                files = list(path.glob("*.txt"))
            else:
                logger.warning(f"No valid .txt files found at {file_path}")
                return

            for txt_file in files:
                logger.info(f"Ingesting documents from {txt_file}")
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()

                chunks = [chunk.strip() for chunk in content.split("---") if chunk.strip()]
                docs = [documents.Document(page_content=chunk) for chunk in chunks]
                ids = [str(uuid4()) for _ in docs]

                self.vector_store.add_documents(documents=docs, ids=ids)
        else:
            logger.info("Skipping ingestion — collection already populated.")

    def needs_rewrite(self, user_input: str) -> bool:
        input_lower = user_input.lower().strip()
        pronouns = ["it", "its", "this", "that", "they", "those", "them", "their"]
        vague_starters = ["what about", "and", "also", "how much", "how long"]

        # Check for pronouns or vague follow-up starters
        if any(p in input_lower.split() for p in pronouns):
            return True
        if any(input_lower.startswith(start) for start in vague_starters):
            return True
        if len(input_lower.split()) < 5 and "?" in input_lower:
            return True

        return False

    def get_all_documents_from_collection(self, collection_name: str, persist_directory: str):
        """
        Load all documents from a persisted Chroma collection using native chromadb client.

        Args:
            collection_name (str): Name of the collection.
            persist_directory (str): Path to persisted Chroma database.

        Returns:
            List[dict]: List of {'id': ..., 'document': ..., 'metadata': ...}
        """
        client = chromadb.PersistentClient(path=persist_directory)

        collection = client.get_collection(name=collection_name)

        print(f"✅ Collection found: {collection.name}")
        print(f"📦 Total documents: {collection.count()}")

        # Now retrieve all documents (no filters, no includes needed for IDs)
        results = collection.get(include=["documents", "metadatas"])

        docs = []
        for doc_id, doc_text, metadata in zip(results['ids'], results['documents'], results['metadatas']):
            docs.append({
                'id': doc_id,
                'document': doc_text,
                'metadata': metadata
            })

        print(f"✅ Retrieved {len(docs)} documents.")
        return docs
            
    def rewrite_ambiguous_prompt(self, messages: list, new_user_input: str) -> str:
        """Rewrite user input using chat history to resolve ambiguity."""
        prompt = "You are a helpful assistant that rewrites ambiguous user questions based on chat history.\n"
        prompt += "Do not answer the question. Just rewrite it to be self-contained. Be consice.\n\n"
        prompt += "Chat history:\n"
        for m in messages[-4:]:  # last few turns for context
            if m["role"] in ("user", "assistant"):
                prompt += f"{m['role'].capitalize()}: {m['content']}\n"
        prompt += f"\nAmbiguous user input: {new_user_input}\n"
        prompt += "Rewritten version:"

        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()

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
