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

# Try to get API key from Streamlit secrets first (for cloud deployment), then from .env
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = os.getenv("GOOGLE_API_KEY")

# Set the API key as environment variable for Google GenAI
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    st.error("⚠️ GOOGLE_API_KEY not found! Please add it to Streamlit secrets or .env file.")

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
    """Read and extract text content from uploaded PDF files."""
    combined_text = ""
    
    for uploaded_pdf in uploaded_pdfs:
        try:
            uploaded_pdf.seek(0)
            pdf = PdfReader(uploaded_pdf)
            pdf_text = ""
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text and len(text.strip()) > 10:
                        pdf_text += text + "\n"
                except Exception as page_error:
                    st.warning(f"Error on page {page_num + 1}: {str(page_error)}")
            
            if pdf_text.strip():
                combined_text += pdf_text
                st.success(f"✓ Extracted {len(pdf_text)} characters from {uploaded_pdf.name}")
            else:
                st.error(f"✗ {uploaded_pdf.name} appears to be a scanned/image PDF with no extractable text.")
                
        except Exception as e:
            st.error(f"Error reading {uploaded_pdf.name}: {str(e)}")
    
    if not combined_text.strip():
        raise ValueError("No text could be extracted from the PDF files.")
    
    return combined_text

def split_text_into_chunks(full_text):
    """Break down large text into smaller chunks with overlap for context retention."""
    if not full_text or not full_text.strip():
        raise ValueError("Cannot split empty text into chunks")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(full_text)
    
    if not chunks:
        raise ValueError("Text splitting resulted in no chunks")
    
    return chunks

def build_and_save_vector_index(chunks):
    """Generate vector embeddings for text chunks and save them as a FAISS index."""
    if not chunks:
        raise ValueError("No chunks provided for embedding")
    
    try:
        genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_index = FAISS.from_texts(chunks, embedding=genai_embeddings)
        vector_index.save_local("vector_index")
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        raise

def get_conversational_chain():
    """Set up the conversational chain for question answering."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context". Don't provide wrong answers.
    
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def process_user_query(user_query):
    """Search relevant context and generate responses for user queries."""
    try:
        genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("vector_index", genai_embeddings, allow_dangerous_deserialization=True)
        
        # Retrieve relevant documents
        docs = vector_store.similarity_search(user_query, k=10)
        
        # Get conversational chain
        chain = get_conversational_chain()
        
        # Get response
        response = chain({"input_documents": docs, "question": user_query}, return_only_outputs=True)
        
        st.write("**AI Response:**", response["output_text"])
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def application_interface():
    """Define the main interface and workflow of the Streamlit app."""
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

    # Add custom CSS
    add_custom_css()

    # App Header
    st.title("📖 PDF Chat Assistant")
    st.markdown("**Interact with your PDFs effortlessly using advanced AI!**")

    # Multi-tab layout
    tabs = st.tabs(["📂 Upload PDFs", "ℹ️ About"])

    # State variable to toggle the question box
    if "show_question_box" not in st.session_state:
        st.session_state["show_question_box"] = False

    with tabs[0]:  # Upload PDFs Tab
        st.header("📂 Upload and Process PDFs")
        st.markdown("**Supported:** PDF files up to 500MB each")
        uploaded_files = st.file_uploader(
            "Upload your PDF files here:", 
            accept_multiple_files=True,
            type=['pdf'],
            help="You can upload multiple PDF files at once"
        )
        
        # Hide question box if no files are uploaded
        if not uploaded_files:
            st.session_state["show_question_box"] = False
        
        # Show uploaded files
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                file_size_mb = file.size / (1024 * 1024)
                st.write(f"- {file.name} ({file_size_mb:.2f} MB)")

        if st.button("Process PDFs"):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    try:
                        document_text = extract_text_from_pdfs(uploaded_files)
                        text_segments = split_text_into_chunks(document_text)
                        build_and_save_vector_index(text_segments)
                        st.success("PDFs successfully processed!")
                        st.session_state["show_question_box"] = True
                    except Exception as e:
                        st.error(f"Failed to process PDFs: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file.")

        # Add spacing after the Process PDFs button
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        # Display question input box after processing
        if st.session_state["show_question_box"]:
            st.header("💬 Ask Questions from Your PDFs")
            query = st.text_input("Type your question here:")
            if query:
                process_user_query(query)

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
