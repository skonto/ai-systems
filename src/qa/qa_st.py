'''Streamlit application that implements a QA bot.'''

import argparse
import logging
import os
import numpy as np
import ollama
import streamlit as st
from guardrails.hub import ToxicLanguage
from guardrails import Guard
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from PIL import Image
from opik import track, opik_context
import opik
import re

# from sentence_transformers import CrossEncoder
opik.configure(use_local=True, automatic_approvals=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

sp = """
You are a helpful, polite, and knowledgeable customer support assistant for a company that sells chargers and other electronic devices.

Your job is to answer customer questions using only the information provided in the retrieved context and any past user question and answer. Do not invent information.

Your objectives:
- Provide clear, accurate, and friendly responses.
- Ask clarifying questions if the user’s request is vague or missing key details.
- Stick strictly to known facts. Never speculate or make up information.
- If the request is outside your scope (e.g. refunds, legal issues), politely direct the user to human support.

For escalation or human support, refer customers to: **support@chargepro.com**

Tone guidelines:
- Be friendly, professional, and empathetic.
- Match the customer's tone, but never be sarcastic or emotional.
- Prioritize helpfulness, clarity, and honesty.


If the answer is not found in the context or previous interactions, respond with:
"Sorry, I cannot answer that based on the available information."

Use the following context and the user's question to answer, the context is not provided by the user:

Context:
---
Q:
A:
Q:
A:
...
---

User question:
{{user_question}}
"""

if "qas" not in st.session_state: 
    st.session_state.qas = [{"role": "system", "content": sp}]

logging.basicConfig(level=logging.WARNING)
st_logger = logging.getLogger('streamlit')
st_logger.setLevel(logging.WARNING)
logger = logging.getLogger("qa")
logger.setLevel(logging.DEBUG)

def main():
    parser = argparse.ArgumentParser(
        description="Parse a db path from the command line."
    )
    parser.add_argument(
        "--db_path",
        type=str,
        help="The path to the vector store db dir.",
        default="/tmp/ch_db",
    )

    args = parser.parse_args()
    db_path = args.db_path

    if not os.path.isdir(db_path):
        print(f"Error: The db path '{db_path}' does not exist.")
        
    
    model_base_url = os.getenv("OLLAMA_HOST")

    if model_base_url is None:
       model_base_url="127.0.0.1:11434"

    guard = Guard().use(
        ToxicLanguage(use_local=True, threshold=0.5, validation_method="sentence", on_fail="exception")
    )
    
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=model_base_url, num_gpu=0, keep_alive=-1)

    vector_store = Chroma(
        collection_name="qas",
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    st.title("Customer Support- Q&A")
    st.divider()
    path_logo = os.path.dirname(os.path.abspath(__file__)) + "/logo.png"
    im = Image.open(path_logo)
    im = im.resize((50, 50))

    with st.chat_message("assistant", avatar=im):
        st.write("Hello! How can I help?")

    question = st.chat_input("What would you like to ask?")

    for msg in st.session_state.messages:
        if "user" in msg:
            with st.chat_message("user"):
                st.write(msg["user"])
        if "assistant" in msg:
            with st.chat_message("assistant", avatar=im):
                st.write(msg["assistant"])

    if question:
        with st.spinner("Working on your input…"):
            retrieved_context = ""
            empty = False
            if is_collection_empty(vector_store):
                print("I have no data")
                empty = True

            docs = []

            if not empty:
                docs = vector_store.similarity_search_with_relevance_scores(
                    f"{question}", k=5, score_threshold=0.4
                )
                for doc in docs:
                    logger.debug(f"Chunk Score: {doc[1]}\n")
                    logger.debug(f"Chunk: {doc[0].page_content}\n-------\n")
                    retrieved_context += doc[0].page_content
                    
                # model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
                # # re-rank the results with original query and documents returned from Chroma
                # scores = model.predict([(question, doc[0].page_content) for doc in docs])
                # # get the highest scoring document
                # print(scores)

            qas = st.session_state.qas

            response = ""

            try:
                # Test failing response
                validation_outcome = guard.validate(question)
                print(validation_outcome.validation_summaries)
            except Exception as e:
                print(e)
                response = "Your question is not appropriate pls try again"

            llama_prompt = format_prompt(question=question, context=retrieved_context)

            qas = qas + [ {'role': 'user', 'content': llama_prompt} ]

            print(qas)

            if response != "Your question is not appropriate pls try again":
                # if retrieved_context == "":
                #     response = "Question does not seem to be relative to the domain of this application. Please try again!"
                # else:   
                    print(llama_prompt + "\n")
                    #output = ollama.chat(model="llama3.2:3b", keep_alive=-1, messages=qas)
                    print("QAS passed to LLM:")
                    for msg in qas:
                        print(msg)
                    output = ollama_llm_call(qas)
                    response = output["message"]["content"]

            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant", avatar=im):
                out = str(response)
                st.write(out)
            
            st.session_state.qas.append( {'role': 'user', 'content': question})
            st.session_state.qas.append( {'role': 'assistant', 'content': response})
            a = {"user": question, "assistant": str(response)}
            st.session_state.messages.append(a)
            
        
@track(tags=['ollama', 'python-library'])
def ollama_llm_call(msgs):
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

def is_collection_empty(vector_store):
    """
    Checks if the db is empty.

    Args:
        vector_store : the vector store

    Returns:
        bool:  if db is empty
    """

    docs = vector_store.get()["ids"]
    return len(docs) == 0


def format_prompt(question, context):
    """
    Formats the prompt according to RAG and history.

    Args:
        question (str): the question passed by the user
        context (str): the context retrieved via RAG.

    Returns:
        str: formatted prompt.
    """

    inject = ""
    if context != "":
        inject = f"{clean_qa_style_context(context)}"
    return f"""
    Context:
    ---
    {inject}
    ---
    User question:
    {question}
    """

def clean_qa_style_context(raw_context: str) -> str:
    """
    Cleans a Q&A-style context string and formats it as:

    Context:
    ---
    Q: <question>
    A: <answer>
    ...
    ---

    Args:
        raw_context (str): Context with Q&A pairs separated by '---'.

    Returns:
        str: Formatted and cleaned Q&A context.
    """
    qa_blocks = raw_context.strip().split("---")
    formatted_pairs = []

    for block in qa_blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue  # skip if not a proper Q&A pair

        # First non-empty line as question
        question = next((l.strip() for l in lines if l.strip()), "")
        # All subsequent non-question lines as answer
        answer_lines = [
            l.strip() for l in lines[1:] 
            if l.strip() and not re.match(r"^(what|how|when|why|does|do|is|are|can|should|who|where|which|would|will|did)\b.*\?\s*$", l.strip().lower())
        ]

        if question and answer_lines:
            answer = " ".join(answer_lines)
            formatted_pairs.append(f"Q: {question}\nA: {answer}")

    return "\n".join(formatted_pairs)


if __name__ == "__main__":
    main()
