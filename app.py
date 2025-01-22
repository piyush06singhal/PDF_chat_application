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

# Initialize API configuration
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Custom CSS for enhanced UI with dark theme and improved visibility
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
                font-size: 20px !important;
                padding: 12px 24px;
            }
            .stButton>button:hover {
                background-color: #55efc4 !important;
                color: black !important;
            }
            .stTextInput>div>div>input {
                border-radius: 10px !important;
                font-size: 20px !important;
                padding: 12px;
                background-color: #2d3436 !important;
                color: white !important;
                border: 1px solid #ffffff !important;
            }
            h1, h2, h3, h4 {
                color: #00cec9 !important;
                font-size: 28px !important;
            }
            .stTabs>div>div>button {
                font-size: 18px !important;
                background-color: #2d3436 !important;
                color: #ffffff !important;
                border: 1px solid #00cec9 !important;
                border-radius: 5px !important;
            }
            .stTabs>div>div>button:hover {
                background-color: #636e72 !important;
            }
            footer {
                text-align: center;
                color: #dfe6e9;
                font-size: 16px;
                margin-top: 20px;
            }
            .stMarkdown h1 {
                font-size: 32px !important;
                color: #00cec9 !important;
            }
            .stMarkdown p {
                font-size: 18px !important;
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
    st.write("AI Response:", response["output_text"])

def application_interface():
    """Define the main interface and workflow of the Streamlit app."""
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

    # Add custom CSS
    add_custom_css()

    # Multi-tab layout with "About" section first
    tabs = st.tabs(["ℹ️ About", "📂 Upload PDFs", "💬 Chat with PDFs"])

    with tabs[0]:  # About Tab
        st.header("ℹ️ About This Application")
        st.markdown("""
        ## Welcome to the **PDF Chat Assistant**!

        This application allows you to:
        - **Upload PDFs** to extract and process content.
        - **Ask Questions** interactively and get context-aware answers.
        - **Efficient Search** with AI-powered embeddings and FAISS index.

        ### Features:
        - **Multiple PDF Uploads**: Process multiple PDFs at once.
        - **Smart Querying**: Context-based answers from document text.
        - **User-Friendly Interface**: Optimized for a seamless experience.

        Built with ❤️ using **Streamlit**, **LangChain**, and **Google Generative AI**.
        """)

    with tabs[1]:  # Upload PDFs Tab
        st.header("📂 Upload and Process PDFs")
        uploaded_files = st.file_uploader("Upload your PDF files here:", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    document_text = extract_text_from_pdfs(uploaded_files)
                    text_segments = split_text_into_chunks(document_text)
                    build_and_save_vector_index(text_segments)
                    st.success("PDFs successfully processed and indexed!")
            else:
                st.warning("Please upload at least one PDF file.")

    with tabs[2]:  # Chat with PDFs Tab
        st.header("💬 Ask Questions from Your PDFs")
        query = st.text_input("Type your question here:")
        if query:
            asyncio.run(process_user_query(query))

    # Footer
    st.markdown("<footer>© 2025 AI PDF Assistant. All rights reserved.</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    application_interface()
