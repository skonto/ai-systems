from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from pprint import pprint

# Configure these paths
PERSIST_DIR = "/tmp/ch_db"  # Path to your Chroma DB
COLLECTION_NAME = "qas"  # Match what you used when ingesting

# Initialize embedding function (must match the one used during ingestion)
embedding_fn = OllamaEmbeddings(
            model="BGE-M3",
            base_url="127.0.0.1:11434",
            num_gpu=0,
            keep_alive=1,
        )

# Load the existing Chroma DB
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_fn,
    collection_name=COLLECTION_NAME
)

# Access the underlying collection (Chroma client)
collection = db._collection  # WARNING: this is internal/private API

# Fetch all documents (None = no filter, 0 = start, 1000 = max number to retrieve)
results = collection.get(include=["documents"], limit=1000)

print(f"Total documents: {len(results['ids'])}\n")

for i, (doc_id, content) in enumerate(zip(results['ids'], results['documents']), 1):
    print(f"--- Document {i} ---")
    print(f"ID: {doc_id}")
    print("Content:")
    print(content)
    print()