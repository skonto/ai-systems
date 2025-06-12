import argparse
import os
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core import documents
from langchain_ollama import OllamaEmbeddings

from rag import OllamaRag

def main():
    parser = argparse.ArgumentParser(
        description="Parse a file path from the command line."
    )
    parser.add_argument("file_path", type=str, help="The path to the file to process")
    parser.add_argument(
        "db_path", type=str, help="The path to the vector store db dir."
    )

    args = parser.parse_args()
    file_path = args.file_path
    db_path = args.db_path

    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")

    if not os.path.isdir(db_path):
        print(f"Warning: The dir '{db_path}' does not exist but it will be created.")
    
    OllamaRag().ingest_docs(file_path, db_path)


if __name__ == "__main__":
    main()
