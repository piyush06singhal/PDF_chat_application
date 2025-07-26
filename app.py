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

# --- INITIALIZATION ---
# Load environment variables for local development.
# On Streamlit Cloud, secrets are loaded automatically.
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


# --- UI STYLING ---
def add_custom_css():
    """Adds custom CSS for the original dark-themed UI."""
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


# --- CORE LOGIC ---
def extract_text_from_pdfs(uploaded_pdfs):
    """Reads and extracts text from multiple PDF files."""
    combined_text = ""
    for uploaded_pdf in uploaded_pdfs:
        try:
            pdf_reader = PdfReader(uploaded_pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    combined_text += page_text
        except Exception as e:
            st.error(f"Error reading '{uploaded_pdf.name}': {e}")
    return combined_text

def split_text_into_chunks(full_text):
    """Splits large text into smaller chunks for processing."""
    # Reduced chunk size for better performance and to avoid timeouts.
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_text(full_text)

def build_and_save_vector_index(chunks, key):
    """Creates a FAISS vector index, passing the API key directly."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
        vector_index = FAISS.from_texts(chunks, embedding=embeddings)
        vector_index.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"🔴 Failed to create vector index: {e}")
        return False

def get_qa_chain(key):
    """Configures the QA chain, passing the API key directly."""
    prompt_template = """
    You are a helpful assistant. Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "The answer is not available in the context."
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3, google_api_key=key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_user_query(user_query, key):
    """Processes the user's query, passing the API key directly."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        relevant_docs = vector_store.similarity_search(user_query)

        if not relevant_docs:
            st.warning("Could not find relevant information for your query.")
            return

        qa_chain = get_qa_chain(key)
        response = qa_chain({"input_documents": relevant_docs, "question": user_query}, return_only_outputs=True)
        st.write("**AI Response:**", response["output_text"])

    except FileNotFoundError:
        st.error("🔴 Vector index not found. Please process your PDF files first.")
    except Exception as e:
        st.error(f"🔴 An error occurred: {e}")


# --- MAIN APPLICATION INTERFACE ---
def main():
    """Defines the main Streamlit application interface with the original tab layout."""
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")
    add_custom_css()

    st.title("📖 PDF Chat Assistant")
    st.markdown("**Interact with your PDFs effortlessly using advanced AI!**")

    tabs = st.tabs(["📂 Upload PDFs", "ℹ️ About"])

    if "show_question_box" not in st.session_state:
        st.session_state.show_question_box = False

    with tabs[0]:  # Upload PDFs Tab
        st.header("📂 Upload and Process PDFs")
        uploaded_files = st.file_uploader("Upload your PDF files here:", accept_multiple_files=True, type="pdf")

        if st.button("Process PDFs"):
            if uploaded_files:
                # Use st.status for more detailed, non-blocking feedback.
                with st.status("Processing documents...", expanded=True) as status:
                    st.write("Step 1: Extracting text from PDFs...")
                    document_text = extract_text_from_pdfs(uploaded_files)
                    if not document_text.strip():
                        status.update(label="Error: No text found!", state="error", expanded=True)
                        st.error("Could not extract text. Ensure PDFs are not image-based.")
                        st.stop()
                    st.write("✅ Text extracted successfully.")

                    st.write("Step 2: Splitting text into chunks...")
                    text_chunks = split_text_into_chunks(document_text)
                    if not text_chunks:
                        status.update(label="Error: Failed to create chunks!", state="error", expanded=True)
                        st.error("Failed to split documents into chunks.")
                        st.stop()
                    st.write(f"✅ Text split into {len(text_chunks)} chunks.")

                    st.write("Step 3: Creating vector index (this may take a while)...")
                    if build_and_save_vector_index(text_chunks, api_key):
                        status.update(label="Processing complete!", state="complete", expanded=False)
                        st.session_state.show_question_box = True
                    else:
                        status.update(label="Error: Failed during vector creation!", state="error", expanded=True)
                        st.error("An error occurred during vector index creation. Check logs for details.")
            else:
                st.warning("Please upload at least one PDF file.")

        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        if st.session_state.show_question_box:
            st.header("💬 Ask Questions from Your PDFs")
            query = st.text_input("Type your question here:")
            if query:
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

    # Footer
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
    main()
