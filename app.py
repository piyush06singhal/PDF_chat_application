import asyncio
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import OpenAIEmbeddings  # Use OpenAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Generate and save vector store using embeddings."""
    embeddings = OpenAIEmbeddings()  # Using OpenAI Embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

async def get_conversational_chain():
    """Set up a question-answering chain with a custom prompt asynchronously."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say, "Answer is not available in the context."
    Do not provide incorrect or fabricated answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Handle user queries by performing similarity search and generating answers."""
    embeddings = OpenAIEmbeddings()  # Using OpenAI Embeddings for consistency
    # Enable dangerous deserialization for trusted sources
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = asyncio.run(get_conversational_chain())  # Async call
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    """Main Streamlit application function."""
    st.set_page_config(page_title="PDF Chat Application")
    st.header("📄 Interact with Your PDFs")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files and Click on Submit Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)  # Ensure vector store is updated
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

    # User input and response generation
    user_question = st.text_input("Ask a question based on the PDF content")
    if user_question:
        response = user_input(user_question)  # Sync call to avoid async error
        st.write("Reply: ", response)

if __name__ == "__main__":
    main()
