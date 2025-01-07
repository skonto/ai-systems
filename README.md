# Download FAQ data under some path eg. data/serverless
Save in a file named faq.txt

# Install Ollama and pull the models

ollama pull llama3:instruct
olllama pull mxbai-embed-large

# Ingest txt Q&A data in Chroma DB

uv run src/data/faq_ingest.py ./data/serverless/faq.txt /tmp/ch_db

# Run teh Chat Streamlit app

source .venv/bin/activate

python -m streamlit run ./src/data/faq_st.py

Alternativelly:

uv run python -m streamlit run ./src/data/faq_st.py
