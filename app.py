import openai
import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chains import LLMChain
import os

# If you're using environment variables for API key:
openai.api_key = os.getenv("OPENAI_API_KEY")

# If you're using Streamlit secrets, you can use the following:
# openai.api_key = st.secrets["OPENAI_API_KEY"]

# Check if the API key is properly set
if not openai.api_key:
    st.error("OpenAI API key is not set! Please configure your key.")
    raise ValueError("OpenAI API key is missing")

def get_vector_store(text_chunks):
    try:
        embeddings = OpenAIEmbeddings()
        # Create FAISS vector store from text chunks
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except openai.error.AuthenticationError as e:
        st.error(f"Authentication Error: {e}")
        raise  # Re-raise to stop further execution
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        raise  # Re-raise to stop further execution

def main():
    # Example: Replace with your actual logic
    text_chunks = ["This is a sample document.", "Here is another chunk."]
    vector_store = get_vector_store(text_chunks)
    
    if vector_store:
        st.write("Vector store created successfully!")
        # Continue with your logic here...

if __name__ == "__main__":
    main()
