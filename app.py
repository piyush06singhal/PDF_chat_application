import openai
import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import os

# Load the OpenAI API key from environment variable or Streamlit secrets
# For local environment, use the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# If you're using Streamlit Cloud, you can store the API key in secrets.toml and use the following line:
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
    st.title("PDF Chat Application")

    # File upload widget for PDF files
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Read the PDF file
        try:
            # You can add your PDF processing logic here (for example, using PyPDF2 or other PDF libraries)
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_chunks = []

            # Extract text from each page in the PDF and store as chunks
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_chunks.append(text)

            if text_chunks:
                # Create the vector store from the text chunks
                vector_store = get_vector_store(text_chunks)

                if vector_store:
                    st.write("Vector store created successfully!")
                    
                    # Example: You can further integrate this vector store with your chatbot or search system
                    # Implement your chatbot logic or search functionality here

                    st.success("Application is ready for interacting with the uploaded PDF.")
                else:
                    st.error("Failed to create the vector store.")
            else:
                st.warning("No text found in the PDF.")
        
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
    
    else:
        st.info("Please upload a PDF to get started.")

if __name__ == "__main__":
    main()
