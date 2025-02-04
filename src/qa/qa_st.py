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

if "message_list" not in st.session_state:
    st.session_state.message_list = []

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

    for l in st.session_state.message_list:
        if "user" in l:
            with st.chat_message("user"):
                st.write(l["user"])
        if "assistant" in l:
            with st.chat_message("assistant", avatar=im):
                st.write(l["assistant"])

    if question:
        with st.spinner("Processing…"):
            retrieved_context = ""
            history = ""
            empty = False
            if is_collection_empty(vector_store):
                print("I have no data")
                empty = True

            docs = []

            if not empty:
                # docs = vector_store.similarity_search(f"{prompt}", k=1)
                docs = vector_store.similarity_search_with_relevance_scores(
                    f"{question}", k=1, score_threshold=0.4
                )
                # for doc in docs:
                #     print(f'Page: {doc.page_content}\n-------\n')
                #     dout += doc.page_content
                for doc in docs:
                    logger.debug(f"Chunk Score: {doc[1]}\n-------\n")
                    logger.debug(f"Chunk: {doc[0].page_content}\n-------\n")
                    retrieved_context += doc[0].page_content

            msgs = st.session_state.message_list
            # Only add the last msg
            if len(msgs) > 0:
                pop = msgs[-1]
                history = "\n user:" + pop["user"] + "\nassistant: " + pop["assistant"]

            response = ""

            try:
                # Test failing response
                validation_outcome = guard.validate(question)
                print(validation_outcome.validation_summaries)
            except Exception as e:
                print(e)
                response = "Your question is toxic pls try again"

            llama_prompt = format_prompt(question=question, context=retrieved_context, history=history)

            if response != "Your question is toxic pls try again":
                print(llama_prompt + "\n")
                output = ollama.generate(model="llama3:instruct", prompt=llama_prompt)
                response = output["response"]

            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant", avatar=im):
                out = str(response)
                st.write(out)

            a = {"user": question, "assistant": str(response)}

            st.session_state.message_list.append(a)


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


def format_prompt(question, context, history):
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
    if context != "" and history != "":
        inject = f"the context: {context} and the history chat: {history}"
    elif context == "" and history != "":
        inject = f"the history chat: {history}"
    elif context != "" and history == "":
        inject = f"the context: {context}"

    return f"You are an assistant that answers questions from a user. Given {inject}. Answer the question: {question}. Be concise."


if __name__ == "__main__":
    main()


# from guardrails.hub import DetectJailbreak
#from guardrails import Guard

# Setup Guard
#guard = Guard().use(
#    DetectJailbreak
#)
