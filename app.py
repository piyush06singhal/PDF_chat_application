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

# --- INITIALIZATION (Corrected) ---
# Load environment variables for local development.
load_dotenv()

# Configure the Google API key from Streamlit secrets.
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔴 Google API Key not found. Please set it in your Streamlit secrets.")
        st.stop()
except Exception as e:
    st.error(f"🔴 Error loading API Key: {e}")
    st.stop()


# --- UI STYLING (Original) ---
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

# --- CORE LOGIC (Corrected) ---
def extract_text_from_pdfs(uploaded_pdfs):
    """Read and extract text content from uploaded PDF files."""
    combined_text = ""
    for uploaded_pdf in uploaded_pdfs:
        try:
            pdf = PdfReader(uploaded_pdf)
            for page in pdf.pages:
                combined_text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading '{uploaded_pdf.name}': {e}")
    return combined_text

def split_text_into_chunks(full_text):
    """Break down large text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    return splitter.split_text(full_text)

def build_and_save_vector_index(chunks, key):
    """Generate vector embeddings and save them as a FAISS index."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
        vector_index = FAISS.from_texts(chunks, embedding=embeddings)
        vector_index.save_local("vector_index")
        return True
    except Exception as e:
        st.error(f"🔴 Failed to create vector index: {e}")
        return False

def get_qa_chain(key):
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
    # Using the correct, stable model name
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.4, google_api_key=key)
    custom_prompt = PromptTemplate(template=prompt_structure, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=custom_prompt)

def process_user_query(user_query, key):
    """Search relevant context and generate responses for user queries."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
        vector_store = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
        relevant_docs = vector_store.similarity_search(user_query)
        qa_chain = get_qa_chain(key)
        response = qa_chain({"input_documents": relevant_docs, "question": user_query}, return_only_outputs=True)
        st.write("**AI Response:**", response["output_text"])
    except Exception as e:
        st.error(f"🔴 An error occurred while processing your query: {e}")


# --- MAIN INTERFACE (Corrected) ---
def application_interface():
    """Define the main interface and workflow of the Streamlit app."""
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")
    add_custom_css()

    st.title("📖 PDF Chat Assistant")
    st.markdown("**Interact with your PDFs effortlessly using advanced AI!**")

    tabs = st.tabs(["📂 Upload PDFs", "ℹ️ About"])

    if "show_question_box" not in st.session_state:
        st.session_state["show_question_box"] = False

    with tabs[0]:  # Upload PDFs Tab
        st.header("📂 Upload and Process PDFs")
        uploaded_files = st.file_uploader("Upload your PDF files here:", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    document_text = extract_text_from_pdfs(uploaded_files)
                    if not document_text.strip():
                        st.error("Could not extract any text from the uploaded PDFs.")
                        st.stop()

                    text_segments = split_text_into_chunks(document_text)
                    if not text_segments:
                        st.error("Failed to split the document into chunks.")
                        st.stop()

                    if build_and_save_vector_index(text_segments, api_key):
                        st.success("PDFs successfully processed!")
                        st.session_state["show_question_box"] = True
            else:
                st.warning("Please upload at least one PDF file.")

        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        if st.session_state["show_question_box"]:
            st.header("💬 Ask Questions from Your PDFs")
            query = st.text_input("Type your question here:")
            if query:
                # Removed asyncio, which is not needed and can cause issues.
                process_user_query(query, api_key)

    with tabs[1]:  # About Tab
        st.header("ℹ️ About This Application")
        st.markdown(""" 
        This **PDF Chat Assistant** allows you to upload PDF files, process their content, and ask questions interactively.

        **Key Features:**
        - Upload and process multiple PDFs.
        - Use AI to generate context-based answers to your queries.
        - Efficient document search using FAISS.

        Built using Streamlit, LangChain, and Google Generative AI.
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
