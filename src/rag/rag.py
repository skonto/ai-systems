from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from loguru import logger
from opik import track, opik_context
from .system_prompts import format_prompt
import ollama
import os

class OllamaRag():
    collection_name : str = "qas"
    db_path: str = "/tmp/ch_db"
    model_base_url = "127.0.0.1:11434" if os.getenv("OLLAMA_HOST") is None else os.getenv("OLLAMA_HOST") 
    
    embeddings = OllamaEmbeddings(model="BGE-M3", base_url=model_base_url, num_gpu=0, keep_alive=-1)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    @classmethod
    def get_reponse(cls, text: str, msgs) -> str:
        retrieved_context = ""
        empty = False
    
        if cls.is_collection_empty():
            logger.warning("Collection is empty!")
            empty = True

        docs = []

        if not empty:
            docs = cls.vector_store.similarity_search_with_relevance_scores(
                f"{text}", k=5, score_threshold=0.4
            )
            for doc in docs:
                logger.debug(f"Chunk Score: {doc[1]}\n")
                logger.debug(f"Chunk: {doc[0].page_content}\n-------\n")
                retrieved_context += doc[0].page_content

        llama_prompt = format_prompt(question=text, context=retrieved_context)

        qas = msgs + [{'role': 'user', 'content': llama_prompt} ]

        logger.debug("chunks passed to LLM:")
        for msg in qas:
                logger.debug(msg)
        output = cls.ollama_llm_call(qas)
        return output["message"]["content"]

    @classmethod
    def is_collection_empty(cls):
        """
        Checks if the db is empty.

        Args:
            vector_store : the vector store

        Returns:
            bool:  if db is empty
        """

        docs = cls.vector_store.get()["ids"]
        return len(docs) == 0
    
    @classmethod
    @track(tags=['ollama', 'python-library'])
    def ollama_llm_call(cls, msgs):
        # Create the Ollama model
        response = ollama.chat(model="llama3.2:3b", keep_alive=-1, messages=msgs, options={"temperature": 0.0, "seed": 1234, "top_k":1, "num_predict": 400})

        opik_context.update_current_span(
            # https://github.com/ollama/ollama/blob/main/docs/api.md
            metadata={
                'model': response['model'],
                'eval_duration': response['eval_duration'],
                'load_duration': response['load_duration'],
                'prompt_eval_duration': response['prompt_eval_duration'],
                'prompt_eval_count': response['prompt_eval_count'],
                'done': response['done'],
                'done_reason': response['done_reason'],
            },
            usage={
                'completion_tokens': response['eval_count'],
                'prompt_tokens': response['prompt_eval_count'],
                'total_tokens': response['eval_count'] + response['prompt_eval_count']
            }
        )
        return response
