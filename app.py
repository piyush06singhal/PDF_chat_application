import asyncio
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time  # For smooth UI animations

# Initialize API configuration
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Custom CSS for enhanced UI with black background
def add_custom_css():
    st.markdown(
        """
        <style>
            body {
                background-color: black;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stApp {
                background: black;
                color: #ffffff;
            }
            .stButton>button {
                background-color: #00b894 !important;
                color: white !important;
                border-radius: 10px !important;
                font-size: 18px !important;
                padding: 10px 20px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
            }
            .stButton>button:hover {
                background-color: #55efc4 !important;
                color: black !important;
                transform: scale(1.05);
                transition: 0.3s ease;
            }
            .stTextInput>div>div>input {
                border-radius: 10px !important;
                font-size: 18px !important;
                padding: 10px;
                background-color: #2d3436 !important;
                color: white !important;
                border: 1px solid #ffffff !important;
            }
            h1, h2, h3, h4 {
                color: #00cec9 !important;
            }
            .separator {
                border: 0;
                height: 2px;
                background: linear-gradient(90deg, #00cec9, #2d3436, #00cec9);
                margin: 20px 0;
            }
            footer {
                text-align: center;
                color: #dfe6e9;
                font-size: 14px;
                margin-top: 20px;
            }
            footer .footer-icons {
                margin-top: 10px;
            }
            footer .footer-icons a {
                color: #00cec9;
                font-size: 20px;
                margin: 0 10px;
                transition: color 0.3s;
            }
            footer .footer-icons a:hover {
                color: #55efc4;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def extract_text_from_pdfs(uploaded_pdfs):
    """Read and extract text content from uploaded PDF files."""
    combined_text = ""
    for uploaded_pdf in uploaded_pdfs:
        pdf = PdfReader(uploaded_pdf)
        for page in pdf.pages:
            combined_text += page.extract_text()
    return combined_text

def split_text_into_chunks(full_text):
    """Break down large text into smaller chunks with overlap for context retention."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    return splitter.split_text(full_text)

def build_and_save_vector_index(chunks):
    """Generate vector embeddings for text chunks and save them as a FAISS index."""
    genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = FAISS.from_texts(chunks, embedding=genai_embeddings)
    vector_index.save_local("vector_index")

async def configure_qa_chain():
    """Set up the question-answering chain with a customized prompt."""
    prompt_structure = """
    Provide detailed answers based on the context provided. 
    If the information is unavailable, respond with, "The context does not contain the answer."
    Avoid generating inaccurate or fabricated responses.

    Context:
    {context}

    User Query:
    {question}

    Response:
    """
    conversational_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    custom_prompt = PromptTemplate(template=prompt_structure, input_variables=["context", "question"])
    return load_qa_chain(conversational_model, chain_type="stuff", prompt=custom_prompt)

async def process_user_query(user_query):
    """Search relevant context and generate responses for user queries asynchronously."""
    genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("vector_index", genai_embeddings, allow_dangerous_deserialization=True)
    relevant_docs = vector_store.similarity_search(user_query)
    qa_chain = await configure_qa_chain()
    response = qa_chain({"input_documents": relevant_docs, "question": user_query}, return_only_outputs=True)
    st.write("**AI Response:**", response["output_text"])

def application_interface():
    """Define the main interface and workflow of the Streamlit app."""
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

    # Add custom CSS
    add_custom_css()

    # App Header
    st.title("📖 PDF Chat Assistant")
    st.markdown("**Interact with your PDFs effortlessly using advanced AI!**")

    # State variable to toggle the question box
    if "show_question_box" not in st.session_state:
        st.session_state["show_question_box"] = False

    st.markdown("<hr class='separator'>", unsafe_allow_html=True)

    # PDF Upload Section
    st.header("📂 Upload and Process PDFs")
    uploaded_files = st.file_uploader("Upload your PDF files here:", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                document_text = extract_text_from_pdfs(uploaded_files)
                text_segments = split_text_into_chunks(document_text)
                build_and_save_vector_index(text_segments)
                time.sleep(1)  # Simulate smooth transition
                st.success("PDFs successfully processed and indexed!")
                st.session_state["show_question_box"] = True
        else:
            st.warning("Please upload at least one PDF file.")
            st.session_state["show_question_box"] = False

    # Display question input box only if PDFs are processed
    if st.session_state["show_question_box"]:
        st.markdown("<hr class='separator'>", unsafe_allow_html=True)
        st.header("💬 Ask Questions from Your PDFs")
        query = st.text_input("Type your question here:")
        if query:
            asyncio.run(process_user_query(query))
    else:
        st.session_state["show_question_box"] = False

    st.markdown("<hr class='separator'>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <footer>
            © 2025 Piyush Singhal. All rights reserved.
            <div class="footer-icons">
                <a href="https://github.com" target="_blank">GitHub</a>
                <a href="https://linkedin.com" target="_blank">LinkedIn</a>
                <a href="https://twitter.com" target="_blank">Twitter</a>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    application_interface()
