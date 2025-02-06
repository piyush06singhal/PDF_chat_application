import streamlit as st
import os
import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Validate API key
if not api_key:
    st.error("❌ API Key is missing. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

# Configure Google AI API
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to configure API Key: {e}")
    st.stop()

# Custom CSS for UI
def add_custom_css():
    st.markdown(
        """
        <style>
            body { background-color: black; color: white; }
            .stApp { background: black; color: white; }
            .stButton>button {
                background-color: #00b894 !important;
                color: white !important;
                border-radius: 10px !important;
                font-size: 18px !important;
                padding: 10px 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def extract_text_from_pdfs(uploaded_pdfs):
    """Extract text from uploaded PDFs."""
    combined_text = ""
    for uploaded_pdf in uploaded_pdfs:
        pdf = PdfReader(uploaded_pdf)
        for page in pdf.pages:
            combined_text += page.extract_text() or ""
    return combined_text

def split_text_into_chunks(full_text):
    """Split text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    return splitter.split_text(full_text)

def build_and_save_vector_index(chunks):
    """Generate vector embeddings and save as FAISS index."""
    genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = FAISS.from_texts(chunks, embedding=genai_embeddings)
    vector_index.save_local("vector_index")

def configure_qa_chain():
    """Configure the question-answering chain."""
    prompt_structure = """
    Context:
    {context}

    User Query:
    {question}

    Response:
    """
    conversational_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    custom_prompt = PromptTemplate(template=prompt_structure, input_variables=["context", "question"])
    return load_qa_chain(conversational_model, chain_type="stuff", prompt=custom_prompt)

def process_user_query(user_query):
    """Retrieve relevant context and generate a response."""
    genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("vector_index", genai_embeddings, allow_dangerous_deserialization=True)
    relevant_docs = vector_store.similarity_search(user_query)
    
    qa_chain = configure_qa_chain()
    response = qa_chain({"input_documents": relevant_docs, "question": user_query}, return_only_outputs=True)
    
    st.write("**AI Response:**", response["output_text"])

def application_interface():
    """Main Streamlit app UI."""
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

    # Add custom CSS
    add_custom_css()

    # App Header
    st.title("📖 PDF Chat Assistant")
    st.markdown("**Upload PDFs and chat with AI to get insights!**")

    # Tabs for Upload & About
    tabs = st.tabs(["📂 Upload PDFs", "ℹ️ About"])

    with tabs[0]:  # Upload PDFs Tab
        st.header("📂 Upload and Process PDFs")
        uploaded_files = st.file_uploader("Upload PDFs:", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    document_text = extract_text_from_pdfs(uploaded_files)
                    text_segments = split_text_into_chunks(document_text)
                    build_and_save_vector_index(text_segments)
                    st.success("✅ PDFs successfully processed!")
                    st.session_state["show_question_box"] = True
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

        if st.session_state.get("show_question_box", False):
            st.header("💬 Ask Questions from Your PDFs")
            query = st.text_input("Type your question here:")
            if query:
                process_user_query(query)

    with tabs[1]:  # About Tab
        st.header("ℹ️ About This Application")
        st.markdown("""
        **PDF Chat Assistant** allows you to upload PDFs, extract text, and ask AI-powered questions.

        **Features:**
        - Upload & process multiple PDFs
        - AI-based answers using Google Gemini API
        - Fast document search using FAISS
        """)

    # Footer
    st.markdown(
        """
        <footer>
            <p>© 2025 Piyush Singhal. All rights reserved.</p>
            <div>
                <a href="https://github.com/piyush06singhal" target="_blank">GitHub</a> |
                <a href="https://www.linkedin.com/in/piyush--singhal/" target="_blank">LinkedIn</a>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    application_interface()
