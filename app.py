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

# Custom CSS for navy-blue background and adjusted text colors
def add_custom_css():
    st.markdown(
        """
        <style>
            body {
                background-color: #1a1a2e;
                color: #f5f5f5;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stApp {
                background-color: #1a1a2e;
                color: #f5f5f5;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #f5f5f5;
            }
            p, li, ul {
                color: #dcdcdc;
            }
            .stButton>button {
                background-color: #e94560 !important;
                color: white !important;
                border-radius: 10px !important;
                font-size: 18px !important;
                padding: 10px 20px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
            }
            .stButton>button:hover {
                background-color: #ef798a !important;
                color: black !important;
            }
            .stTextInput>div>div>input {
                background-color: #0f3460 !important;
                color: white !important;
                border-radius: 10px !important;
                font-size: 16px !important;
                padding: 10px;
                border: 1px solid #e94560;
            }
            .stTextInput>div>div>input::placeholder {
                color: #d3d3d3 !important;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 18px !important;
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Function to extract text from PDFs
def extract_text_from_pdfs(uploaded_pdfs):
    """Read and extract text content from uploaded PDF files."""
    combined_text = ""
    for uploaded_pdf in uploaded_pdfs:
        pdf = PdfReader(uploaded_pdf)
        for page in pdf.pages:
            combined_text += page.extract_text()
    return combined_text

# Function to split text into chunks
def split_text_into_chunks(full_text):
    """Break down large text into smaller chunks with overlap for context retention."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    return splitter.split_text(full_text)

# Function to build and save vector index
def build_and_save_vector_index(chunks):
    """Generate vector embeddings for text chunks and save them as a FAISS index."""
    genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = FAISS.from_texts(chunks, embedding=genai_embeddings)
    vector_index.save_local("vector_index")

# Asynchronous function to configure QA chain
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

# Asynchronous function to process user query
async def process_user_query(user_query):
    """Search relevant context and generate responses for user queries asynchronously."""
    genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("vector_index", genai_embeddings, allow_dangerous_deserialization=True)
    relevant_docs = vector_store.similarity_search(user_query)
    qa_chain = await configure_qa_chain()
    response = qa_chain({"input_documents": relevant_docs, "question": user_query}, return_only_outputs=True)
    st.write("**AI Response:**", response["output_text"])

# Application interface
def application_interface():
    """Define the main interface and workflow of the Streamlit app."""
    add_custom_css()  # Add custom styling

    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

    # Multi-tab layout
    tabs = st.tabs(["ℹ️ About", "📂 Upload PDFs", "💬 Chat with PDFs"])

    with tabs[0]:  # About Tab
        st.header("ℹ️ About This Application")
        st.markdown(
            """
            **Welcome to the PDF Chat Assistant!**  
            
            This tool allows you to:  
            - **Upload and process PDF documents** to extract their content.  
            - **Ask questions interactively** based on the document content.  
            - **Receive detailed answers** powered by AI.  
            
            **Features:**  
            - Multi-document support.  
            - Advanced context-based question answering.  
            - Powered by LangChain, Streamlit, and Google Generative AI.  
            
            Built with ❤️ for seamless document exploration.
            """
        )

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

# Run the app
if __name__ == "__main__":
    application_interface()
