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
# On Streamlit Cloud, secrets are loaded automatically from the secrets manager.
load_dotenv()

# Configure the Google API key.
# The app will fail gracefully if the key is not found.
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔴 Google API Key not found. Please set it in your secrets.")
        st.stop()
    # This line is crucial for the LangChain library to find the key.
    os.environ["GOOGLE_API_KEY"] = api_key
except Exception as e:
    st.error(f"🔴 Error loading API Key: {e}")
    st.stop()


# --- UI STYLING ---
def add_custom_css():
    """Adds custom CSS for a dark-themed, polished UI."""
    st.markdown(
        """
        <style>
            /* General body and app styling */
            body, .stApp {
                background-color: #000000;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            /* Button styling */
            .stButton>button {
                background-color: #00b894 !important;
                color: white !important;
                border-radius: 10px !important;
                font-size: 16px !important;
                padding: 10px 24px;
                border: none;
                transition: background-color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #55efc4 !important;
                color: black !important;
            }

            /* Text input styling */
            .stTextInput>div>div>input {
                border-radius: 10px !important;
                font-size: 16px !important;
                padding: 12px;
                background-color: #2d3436 !important;
                color: white !important;
                border: 1px solid #ffffff !important;
            }

            /* Headers */
            h1, h2, h3, h4 {
                color: #00cec9 !important;
            }

            /* Footer styling */
            footer {
                text-align: center;
                color: #dfe6e9;
                font-size: 14px;
                margin-top: 40px;
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(full_text)

def build_and_save_vector_index(chunks):
    """Creates a FAISS vector index from text chunks and saves it."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_index = FAISS.from_texts(chunks, embedding=embeddings)
        vector_index.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"🔴 Failed to create vector index: {e}")
        return False

def get_qa_chain():
    """Configures and returns a question-answering chain."""
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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_user_query(user_query):
    """Processes the user's query against the vector index."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # The 'allow_dangerous_deserialization' is required for loading FAISS indexes.
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        relevant_docs = vector_store.similarity_search(user_query)

        if not relevant_docs:
            st.warning("Could not find relevant information in the documents for your query.")
            return

        qa_chain = get_qa_chain()
        response = qa_chain({"input_documents": relevant_docs, "question": user_query}, return_only_outputs=True)
        st.write("### Answer")
        st.write(response["output_text"])

    except FileNotFoundError:
        st.error("🔴 Vector index not found. Please upload and process your PDF files first.")
    except Exception as e:
        st.error(f"🔴 An error occurred while processing your query: {e}")


# --- MAIN APPLICATION INTERFACE ---
def main():
    """Defines the main Streamlit application interface."""
    st.set_page_config(page_title="PDF Chat Assistant", page_icon="📖", layout="wide")
    add_custom_css()

    st.title("📖 PDF Chat Assistant")
    st.markdown("#### Interact with your documents using the power of Google's Gemini AI!")

    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.header("Controls")
        uploaded_files = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing... This may take a moment."):
                    document_text = extract_text_from_pdfs(uploaded_files)

                    if not document_text.strip():
                        st.error("Could not extract text. Ensure PDFs are not image-based.")
                        st.stop()

                    text_chunks = split_text_into_chunks(document_text)

                    if not text_chunks:
                        st.error("Failed to split documents into chunks.")
                        st.stop()

                    if build_and_save_vector_index(text_chunks):
                        st.session_state.processed = True
                        st.success("✅ Documents processed successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

    st.markdown("---")

    if st.session_state.processed:
        st.header("💬 Ask a Question")
        user_question = st.text_input("What would you like to know from your documents?")
        if user_question:
            process_user_query(user_question)
    else:
        st.info("Please upload and process your documents using the sidebar to begin.")

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
