import streamlit as st
from langchain_ollama import OllamaEmbeddings
import ollama
from langchain_chroma import Chroma
from glob import glob
import os
from PIL import Image
import argparse

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def is_collection_empty(vectorstore):
    docs = vectorstore.get()["ids"]
    return len(docs) == 0

vector_store = Chroma(
    collection_name="serverless",
    embedding_function=embeddings,
    persist_directory="/tmp/ch_db")

if 'message_list' not in st.session_state:
  st.session_state.message_list = []
  
def main():
  parser = argparse.ArgumentParser(description="Parse a db path from the command line.")
  parser.add_argument(
     "db_path",
     type=str,
     help="The path to the vector store db dir."
     )
    
  args = parser.parse_args()
  db_path = args.db_path

  if not os.path.isdir(db_path):
    print(f"Error: The db path '{db_path}' does not exist.")

  st.title('Opensift Serverless - Q&A Assistant')
  st.divider()
  path_logo = os.path.dirname(os.path.abspath(__file__)) + "/logo.png"
  im = Image.open(path_logo)
  im = im.resize((50,50))
  with st.chat_message('assistant', avatar=im):
    st.write("Hello! How can I help?")
  
  prompt = st.chat_input("Ask a question")
    
  for l in st.session_state.message_list:
    if 'user' in l:
       with st.chat_message("user"):
          st.write(l['user'])
    if 'assistant' in l:
        with st.chat_message("assistant", avatar=im):
          st.write(l['assistant'])

  if prompt:
      with st.spinner('Processing…'):
        dout = ""
        history = ""
        
        if is_collection_empty(vector_store):
           st.write("I have no data")

        # docs = vector_store.similarity_search(f"{prompt}", k=1)
        docs = vector_store.similarity_search_with_relevance_scores(f"{prompt}", k=1, score_threshold=0.4)
        # for doc in docs:
        #     print(f'Page: {doc.page_content}\n-------\n')
        #     dout += doc.page_content
        for doc in docs:
            print(f'Chunk Score: {doc[1]}\n-------\n')
            print(f'Chunk: {doc[0].page_content}\n-------\n')
            dout += doc[0].page_content

        msgs = st.session_state.message_list
        # Only add the last msg
        if len(msgs) > 0:
           pop = msgs[-1]
           history = "\n user:" + pop['user'] + "\nassistant: " + pop['assistant']
        

        llamaPrompt = f"You are an assistant that answers questions from a user. Given the context: {dout} and the history chat: {history}. Answer the question: {prompt}. Be concise."
        print(llamaPrompt + "\n")
        output = ollama.generate(
        model="llama3:instruct",
        prompt=llamaPrompt)
        response = output['response']

        with st.chat_message("user"):
          st.write(prompt)
        with st.chat_message("assistant", avatar=im):
          out = str(response)
          st.write(out)
      
        a = {
          "user": prompt,
          "assistant": str(response)
        }
      
        st.session_state.message_list.append(a)

if __name__ == "__main__":
    main()