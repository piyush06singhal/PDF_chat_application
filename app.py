import asyncio
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load API keys from environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Custom CSS for enhanced UI (unchanged)
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
            }
            .stButton>button:hover {
                background-color: #55efc4 !important;
                color: black !important;
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
            .stTabs>div>div>button {
                font-size: 16px !important;
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
                font-size: 14px;
                margin-top: 20px;
            }
            .footer-links {
                margin-top: 10px;
            }
            .footer-links a {
                color: #00cec9 !important;
                text-decoration: none;
                margin: 0 10px;
            }
            .footer-links a:hover {
                text-decoration: underline;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def extract_text_from_pdfs(uploaded_pdfs):
    """Read and extract text from uploaded PDF files."""
    combined_text = ""
    for uploaded_pdf in uploaded_pdfs:
        pdf = PdfReader(uploaded_pdf)
        for page in pdf.pages:
            combined_text += page.extract_text() or ""
    return combined_text

def split_text_into_chunks(full_text):
    """Break down large text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    return splitter.split_text(full_text)

def build_and_save_vector_index(chunks):
    """Generate vector embeddings using OpenAI and save them in FAISS."""
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    vector_index = FAISS.from_texts(chunks, embedding=embeddings)
    vector_index.save_local("vector_index")

async def configure_qa_chain():
    """Set up the QA chain with OpenAI's GPT-4 model."""
    prompt_structure = """
    Provide a well-structured, concise, and contextually accurate answer.
    If the information is not available, say: "The context does not contain the answer."

    Context:
    {context}

    User Query:
    {question}

    Response:
    """
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.4, openai_api_key=openai_api_key)
    custom_prompt = PromptTemplate(template=prompt_structure, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm, retriever=None, chain_type="stuff", chain_prompt=custom_prompt)

async def process_user_query(user_query):
    """Search relevant context and generate responses asynchronously."""
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    vector_store = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vector_store.as_retriever()
    qa_chain = await configure_qa_chain()
    qa_chain.retriever = retriever

    response = qa_chain.run({"query": user_query})
    st.write("**AI Response:**", response)

def application_interface():
    """Define the main interface and workflow of the Streamlit app."""
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

    # Add custom CSS
    add_custom_css()

    # App Header
    st.title("📖 PDF Chat Assistant")
    st.markdown("**Interact with your PDFs effortlessly using AI!**")

    # Multi-tab layout
    tabs = st.tabs(["📂 Upload PDFs", "ℹ️ About"])

    # State variable to toggle the question box
    if "show_question_box" not in st.session_state:
        st.session_state["show_question_box"] = False

    with tabs[0]:  # Upload PDFs Tab
        st.header("📂 Upload and Process PDFs")
        uploaded_files = st.file_uploader("Upload your PDF files here:", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    document_text = extract_text_from_pdfs(uploaded_files)
                    text_segments = split_text_into_chunks(document_text)
                    build_and_save_vector_index(text_segments)
                    st.success("PDFs successfully processed! Now you can ask questions.")
                    st.session_state["show_question_box"] = True
            else:
                st.warning("Please upload at least one PDF file.")

        # Display question input box after processing
        if st.session_state["show_question_box"]:
            st.header("💬 Ask Questions from Your PDFs")
            query = st.text_input("Type your question here:")
            if query:
                asyncio.run(process_user_query(query))

    with tabs[1]:  # About Tab
        st.header("ℹ️ About This Application")
        st.markdown(""" 
        This **PDF Chat Assistant** allows you to upload PDF files, process their content, and ask questions interactively.

        **Key Features:**
        - Upload and process multiple PDFs.
        - Use AI to generate context-based answers to your queries.
        - Efficient document search using FAISS.

        Built using Streamlit, LangChain, FAISS, and OpenAI GPT-4.
        """)

    # Footer with social media links
    st.markdown(
        """
        <footer>
            <p>© 2025 Piyush Singhal. All rights reserved.</p>
            <div class="footer-links">
                <a href="https://github.com/piyush06singhal" target="_blank">GitHub</a> |
                <a href="https://www.linkedin.com/in/piyush--singhal/" target="_blank">LinkedIn</a> |
                <a href="https://x.com/PiyushS07508112" target="_blank">Twitter</a>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    application_interface()
