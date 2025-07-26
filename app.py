import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.Youtubeing import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables for local development.
# On Streamlit Cloud, secrets are loaded automatically.
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def add_custom_css():
    """Adds custom CSS for styling the Streamlit app."""
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
    """Reads and extracts text content from uploaded PDF files."""
    combined_text = ""
    for uploaded_pdf in uploaded_pdfs:
        try:
            pdf_reader = PdfReader(uploaded_pdf)
            for page in pdf_reader.pages:
                # Add text from page, ensuring it's not None
                page_text = page.extract_text()
                if page_text:
                    combined_text += page_text
        except Exception as e:
            st.error(f"Error reading '{uploaded_pdf.name}': {e}")
    return combined_text

def split_text_into_chunks(full_text):
    """Breaks down large text into smaller, manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    return splitter.split_text(full_text)

def build_and_save_vector_index(chunks):
    """Generates vector embeddings for text chunks and saves them locally."""
    genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = FAISS.from_texts(chunks, embedding=genai_embeddings)
    vector_index.save_local("vector_index")

def configure_qa_chain():
    """Sets up the question-answering chain with a customized prompt."""
    prompt_structure = """
    Provide a detailed and comprehensive answer based strictly on the provided context.
    If the information required to answer the question is not in the context, clearly state:
    "The context does not contain the answer to this question."
    Do not invent, assume, or use external knowledge.

    Context:
    {context}

    User Query:
    {question}

    Response:
    """
    conversational_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    custom_prompt = PromptTemplate(template=prompt_structure, input_variables=["context", "question"])
    return load_qa_chain(conversational_model, chain_type="stuff", prompt=custom_prompt)

def process_user_query(user_query):
    """Searches relevant context and generates a response for the user's query."""
    genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.load_local("vector_index", genai_embeddings, allow_dangerous_deserialization=True)
        relevant_docs = vector_store.similarity_search(user_query)
        
        if not relevant_docs:
            st.warning("Could not find relevant documents in the PDF for your query.")
            return

        qa_chain = configure_qa_chain()
        response = qa_chain({"input_documents": relevant_docs, "question": user_query}, return_only_outputs=True)
        st.write("**AI Response:**", response["output_text"])
    except FileNotFoundError:
        st.error("Vector index not found. Please process your PDF files first.")
    except Exception as e:
        st.error(f"An error occurred while processing your query: {e}")

def application_interface():
    """Defines the main user interface and workflow of the Streamlit app."""
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")
    add_custom_css()

    st.title("📖 PDF Chat Assistant")
    st.markdown("**Interact with your PDFs effortlessly using advanced AI!**")

    tabs = st.tabs(["📂 Upload & Chat", "ℹ️ About"])

    if "show_question_box" not in st.session_state:
        st.session_state.show_question_box = False

    with tabs[0]:  # Upload & Chat Tab
        st.header("1. Upload and Process Your PDFs")
        uploaded_files = st.file_uploader("Upload your PDF files here:", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if uploaded_files:
                with st.spinner("Processing PDFs... This may take a moment."):
                    document_text = extract_text_from_pdfs(uploaded_files)
                    
                    if not document_text.strip():
                        st.error("Could not extract any text. Please ensure your PDFs contain selectable text and are not just images.")
                        st.stop()

                    text_segments = split_text_into_chunks(document_text)
                    
                    if not text_segments:
                        st.error("Failed to split the document into processable chunks. The document might be too small.")
                        st.stop()

                    build_and_save_vector_index(text_segments)
                    st.success("✅ PDFs successfully processed! You can now ask questions below.")
                    st.session_state.show_question_box = True
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        if st.session_state.show_question_box:
            st.header("2. Ask Questions About Your PDFs")
            query = st.text_input("Type your question here and press Enter:", key="query_input")
            if query:
                process_user_query(query)

    with tabs[1]:  # About Tab
        st.header("ℹ️ About This Application")
        st.markdown("""
        This **PDF Chat Assistant** allows you to upload one or more PDF files, processes their content using AI, and lets you ask questions about them interactively.

        **Key Features:**
        - **Multiple PDF Upload:** Upload and combine information from several documents.
        - **Contextual AI Answers:** Uses Google's Gemini model to generate answers based *only* on the content of your documents.
        - **Efficient Document Search:** Powered by FAISS vector search for fast and relevant context retrieval.

        Built using **Streamlit**, **LangChain**, and **Google Generative AI**.
        """)

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
