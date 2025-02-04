import asyncio
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Initialize API configuration
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Custom CSS for enhanced UI with black background
def add_custom_css():
    st.markdown(
        """
        <style>
            body { background-color: black; color: #ffffff; }
            .stApp { background: black; color: #ffffff; }
            .stButton>button { background-color: #00b894 !important; color: white !important; }
            .stButton>button:hover { background-color: #55efc4 !important; color: black !important; }
            .stTextInput>div>div>input { background-color: #2d3436 !important; color: white !important; }
            h1, h2, h3, h4 { color: #00cec9 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def extract_text_from_pdfs(uploaded_pdfs):
    combined_text = ""
    for uploaded_pdf in uploaded_pdfs:
        pdf = PdfReader(uploaded_pdf)
        for page in pdf.pages:
            combined_text += page.extract_text() or ""
    return combined_text

def split_text_into_chunks(full_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    return splitter.split_text(full_text)

def build_and_save_vector_index(chunks):
    embeddings = OpenAIEmbeddings()
    vector_index = FAISS.from_texts(chunks, embedding=embeddings)
    vector_index.save_local("vector_index")

def configure_qa_chain():
    prompt_structure = """
    Provide detailed answers based on the context provided.
    If the information is unavailable, respond with, "The context does not contain the answer."
    
    Context:
    {context}
    
    User Query:
    {question}
    
    Response:
    """
    conversational_model = ChatOpenAI(temperature=0.4)
    custom_prompt = PromptTemplate(template=prompt_structure, input_variables=["context", "question"])
    return load_qa_chain(conversational_model, chain_type="stuff", prompt=custom_prompt)

async def process_user_query(user_query):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    relevant_docs = vector_store.similarity_search(user_query)
    qa_chain = configure_qa_chain()
    response = qa_chain({"input_documents": relevant_docs, "question": user_query}, return_only_outputs=True)
    st.write("**AI Response:**", response["output_text"])

def application_interface():
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")
    add_custom_css()
    st.title("📖 PDF Chat Assistant")
    st.markdown("**Interact with your PDFs effortlessly using advanced AI!**")
    tabs = st.tabs(["📂 Upload PDFs", "ℹ️ About"])
    if "show_question_box" not in st.session_state:
        st.session_state["show_question_box"] = False
    with tabs[0]:
        st.header("📂 Upload and Process PDFs")
        uploaded_files = st.file_uploader("Upload your PDF files here:", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    document_text = extract_text_from_pdfs(uploaded_files)
                    text_segments = split_text_into_chunks(document_text)
                    build_and_save_vector_index(text_segments)
                    st.success("PDFs successfully processed!")
                    st.session_state["show_question_box"] = True
            else:
                st.warning("Please upload at least one PDF file.")
        if st.session_state["show_question_box"]:
            st.header("💬 Ask Questions from Your PDFs")
            query = st.text_input("Type your question here:")
            if query:
                asyncio.run(process_user_query(query))
    with tabs[1]:
        st.header("ℹ️ About This Application")
        st.markdown(""" 
        This **PDF Chat Assistant** allows you to upload PDF files, process their content, and ask questions interactively.
        **Key Features:**
        - Upload and process multiple PDFs.
        - Use AI to generate context-based answers to your queries.
        - Efficient document search using FAISS.
        Built using Streamlit, LangChain, and OpenAI.
        """)
    st.markdown(
        """
        <footer>
            <p>© 2025 Piyush Singhal. All rights reserved.</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    application_interface()
