from langchain_ollama import OllamaEmbeddings
from langchain_core import documents
from langchain_chroma import Chroma
from uuid import uuid4
import argparse
import os


def is_collection_empty(vectorstore):
    docs = vectorstore.get()["ids"]
    return len(docs) == 0


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

    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")

    db_path = args.db_path
    embeddings = OllamaEmbeddings(model="BGE-M3")

    vector_store = Chroma(
        collection_name="qas",
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    chunks = []
    docs = []

    if is_collection_empty(vector_store):
        print(f"Parsing doc... {file_path}\n")
        with open(file_path, "r") as file:
            file_text = file.read()
            chunks = file_text.split("---")
            for i, c in enumerate(chunks):
                if len(c) != 0:
                    docs.append(documents.Document(page_content=c))

            uuids = [str(uuid4()) for _ in range(len(docs))]
            vector_store.add_documents(documents=docs, ids=uuids)


if __name__ == "__main__":
    main()
