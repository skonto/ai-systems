'''Streamlit application that implements a QA bot.'''

import argparse
import logging
import os
import ollama
import streamlit as st
from guardrails.hub import ToxicLanguage
from guardrails import Guard
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from PIL import Image

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "qas" not in st.session_state: 
    st.session_state.qas = []

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

    guard = Guard().use(
        ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
    )
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vector_store = Chroma(
        collection_name="serverless",
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    st.title("Fantastic Charge - Q&A Assistant")
    st.divider()
    path_logo = os.path.dirname(os.path.abspath(__file__)) + "/logo.png"
    im = Image.open(path_logo)
    im = im.resize((50, 50))

    with st.chat_message("assistant", avatar=im):
        st.write("Hello! How can I help?")

    question = st.chat_input("Ask a question")

    for msg in st.session_state.messages:
        if "user" in msg:
            with st.chat_message("user"):
                st.write(msg["user"])
        if "assistant" in msg:
            with st.chat_message("assistant", avatar=im):
                st.write(msg["assistant"])

    if question:
        with st.spinner("Processing…"):
            retrieved_context = ""
            empty = False
            if is_collection_empty(vector_store):
                print("I have no data")
                empty = True

            docs = []

            if not empty:
                docs = vector_store.similarity_search_with_relevance_scores(
                    f"{question}", k=2, score_threshold=0.4
                )
                for doc in docs:
                    logger.debug(f"Chunk Score: {doc[1]}\n")
                    logger.debug(f"Chunk: {doc[0].page_content}\n-------\n")
                    retrieved_context += doc[0].page_content

            qas = st.session_state.qas

            response = ""

            try:
                # Test failing response
                validation_outcome = guard.validate(question)
                print(validation_outcome.validation_summaries)
            except Exception as e:
                print(e)
                response = "Your question is toxic pls try again"

            llama_prompt = format_prompt(question=question, context=retrieved_context)

            qas = qas + [ {'role': 'user', 'content': llama_prompt} ]

            print(qas)

            if response != "Your question is toxic pls try again":
                print(llama_prompt + "\n")
                output = ollama.chat(model="llama3:instruct", messages=qas)
                response = output["message"]["content"]

            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant", avatar=im):
                out = str(response)
                st.write(out)

            st.session_state.qas.append( {'role': 'user', 'content': llama_prompt})
            st.session_state.qas.append( {'role': 'assistant', 'content': response})
            a = {"user": question, "assistant": str(response)}
            st.session_state.messages.append(a)

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
        history (str): the history of the current session

    Returns:
        str: formatted prompt.
    """

    inject = ""
    if context != "":
        inject = f"the context: {context}"

    p = "You are an assistant that answers questions from a user."

    return p + f"Given {inject}. Answer the question: {question}. Be concise."


if __name__ == "__main__":
    main()
