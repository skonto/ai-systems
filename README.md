
# Instructions


```
# Download FAQ data in txt format under some path eg. data/serverless
# Save in a file named faq.txt. Example of data:
#
# How do I get support?
# For customer related queries please reach out to support@email.com

# Install Ollama and pull the models
ollama pull llama3:instruct
olllama pull mxbai-embed-large

# Ingest txt Q&A data in Chroma DB
uv run src/data/faq_ingest.py ./data/serverless/faq.txt /tmp/ch_db

# Run teh Chat Streamlit app
source .venv/bin/activate

python -m streamlit run ./src/data/faq_st.py

# Alternativelly:

uv run python -m streamlit run ./src/data/faq_st.py
```

# Intaracting with the Q&A Assistant

