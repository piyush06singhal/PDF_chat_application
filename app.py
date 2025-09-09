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

# Initialize API configuration (supports local .env and Streamlit Cloud secrets)
load_dotenv()

# Prefer Streamlit secrets on cloud; fallback to environment variables locally
api_key = None
try:
    # st.secrets is available on Streamlit Cloud
    if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GOOGLE_API_KEY")

# Ensure downstream libraries can read the key from environment
if api_key:
    # Sanitize in case quotes/spaces were pasted around the key
    api_key = api_key.strip().strip('"').strip("'")
    os.environ["GOOGLE_API_KEY"] = api_key

def validate_api_key() -> bool:
    """Quickly validate the Gemini API key with a tiny embeddings call."""
    try:
        test_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # minimal 1-token-ish input to minimize usage
        _ = test_embeddings.embed_query("ok")
        return True
    except Exception as err:
        st.error("API key validation failed. Please ensure you are using a Gemini API key from AI Studio and that it is pasted without quotes/spaces.")
        st.caption(f"Details: {err}")
        return False

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
        pdf = PdfReader(uploaded_pdf)
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            combined_text += page_text
    return combined_text

def split_text_into_chunks(full_text):
    """Break down large text into smaller chunks with overlap for context retention."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(full_text)

def build_and_save_vector_index(chunks):
    """Generate vector embeddings for text chunks and save them as a FAISS index."""
    genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = FAISS.from_texts(chunks, embedding=genai_embeddings)
    vector_index.save_local("vector_index")

def configure_qa_chain():
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
    conversational_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
    custom_prompt = PromptTemplate(template=prompt_structure, input_variables=["context", "question"])
    return load_qa_chain(conversational_model, chain_type="stuff", prompt=custom_prompt)

def process_user_query(user_query):
    """Search relevant context and generate responses for user queries."""
    try:
        if not os.path.isdir("vector_index"):
            st.warning("Please process PDFs first to build the index.")
            return
        genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("vector_index", genai_embeddings, allow_dangerous_deserialization=True)
        relevant_docs = vector_store.similarity_search(user_query, k=4)
        if not relevant_docs:
            st.warning("No relevant context found in the PDFs for this question.")
            return
        qa_chain = configure_qa_chain()
        response = qa_chain({"input_documents": relevant_docs, "question": user_query}, return_only_outputs=True)
        st.write("**AI Response:**", response.get("output_text", ""))
    except Exception as e:
        st.error(f"Failed to generate answer: {str(e)}")

def application_interface():
    """Define the main interface and workflow of the Streamlit app."""
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

    # Add custom CSS
    add_custom_css()

    # Check API key early and fail fast with a clear message
    if not api_key or not isinstance(api_key, str) or len(api_key.strip()) == 0:
        st.error("GOOGLE_API_KEY is missing. Please create a .env file with GOOGLE_API_KEY=<your_key>.")
        st.stop()
    # Validate by doing a tiny embeddings call; if it fails, stop early
    if not validate_api_key():
        st.stop()

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
        uploaded_files = st.file_uploader("Upload your PDF files here:", type=["pdf"], accept_multiple_files=True)

        if st.button("Process PDFs"):
            if uploaded_files:
                try:
                    with st.spinner("Processing PDFs..."):
                        document_text = extract_text_from_pdfs(uploaded_files)
                        if not document_text.strip():
                            st.warning("No extractable text found in the uploaded PDFs.")
                        text_segments = split_text_into_chunks(document_text)
                        if not text_segments:
                            st.warning("No text segments were created. Please try a different PDF.")
                        else:
                            build_and_save_vector_index(text_segments)
                            st.success("PDFs successfully processed!")
                            # Show question box after processing
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
            if st.button("Get Answer"):
                if not query.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Generating answer..."):
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
