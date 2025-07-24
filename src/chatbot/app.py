import argparse
import os
import sys

import streamlit as st
from loguru import logger
from PIL import Image

from guards import validate_input
from observability import setup_tracing
from rag import OllamaRag, get_initial_chat_state

if "logger_configured" not in st.session_state:

    logger.add(
        "app.log",
        rotation="10 MB",
        retention="7 days",
        backtrace=True,
        diagnose=True,
        level="DEBUG",
    )

    st.session_state.logger_configured = True

setup_tracing()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qas" not in st.session_state:
    st.session_state.qas = get_initial_chat_state()

def main() -> None:
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

    st.title("Customer Support- Q&A")
    st.divider()
    path_logo = os.path.dirname(os.path.abspath(__file__)) + "/logo.png"
    image_file = Image.open(path_logo)
    image = image_file.resize((50, 50))

    with st.chat_message("assistant", avatar=image):
        st.write("Hello! How can I help?")

    user_input = st.chat_input("What would you like to ask today?")

    for msg in st.session_state.messages:
        if "user" in msg:
            with st.chat_message("user"):
                st.write(msg["user"])
        if "assistant" in msg:
            with st.chat_message("assistant", avatar=image):
                st.write(msg["assistant"])

    if user_input:
        with st.spinner("Processing…"):
            response = ""

            try:
                validate_input(user_input)
            except ValueError as e:
                response = str(e)

            if response == "":
                qas = st.session_state.qas
                response, _ = OllamaRag().get_response(user_input, qas)

            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant", avatar=image):
                out = str(response)
                st.write(out)

            st.session_state.qas.append({"role": "user", "content": user_input})
            st.session_state.qas.append({"role": "assistant", "content": response})
            a = {"user": user_input, "assistant": str(response)}
            st.session_state.messages.append(a)


if __name__ == "__main__":
    main()
