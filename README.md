## PDF Chat Application
    Interact with your PDF files in a conversational way! This Streamlit-based        application allows you to upload PDFs, process their content, and ask       
    questions about the text extracted from the documents. Using advanced AI 
    models   from Google Generative AI, this tool provides detailed, accurate, 
    and context-aware responses.
``````````````````````````````````````````````````````````````````````````````````
# Features
1. **PDF Upload and Processing:**
    Upload multiple PDF files.
    Extract text from PDFs efficiently using PyPDF2.
   
2. **Text Chunking:**
    Split large text into manageable chunks for better context handling.
   
3. **AI-Powered Q&A:**
    Leverage Google Generative AI (gemini-pro) for question-answering.
    Responses are generated based on the uploaded PDF content.
   
4. **Vector Embedding and Search:**
    Use FAISS to create and store vector embeddings for similarity-based context      retrieval.
   
5. **Streamlit Interface:**
    User-friendly interface for PDF upload, processing, and querying.
``````````````````````````````````````````````````````````````````````````````````
# How It Works
1. **Upload PDFs:**
    Drag and drop one or more PDF files into the app.
   
2. **Text Extraction:**
    Text is extracted from the uploaded PDFs using PyPDF2.
   
3. **Text Chunking:**
    The extracted text is split into smaller chunks using LangChain's           
    RecursiveCharacterTextSplitter to ensure efficient context handling.
   
4. **Vector Store Creation:**
    Text chunks are converted into vector embeddings using Google Generative AI.
    These embeddings are stored in a FAISS index for similarity search.
   
5. **Question-Answering:**
    Users type a question based on the content of the uploaded PDFs.
    Relevant chunks are retrieved using similarity search and passed to Google's 
    Generative AI model to generate a detailed response.
   
6. **Response Display:**
    The app displays the AI's response in a clear and concise manner.
``````````````````````````````````````````````````````````````````````````````````
# Tech Stack
1. **Frontend:**
    Streamlit: Interactive user interface for PDF upload and Q&A.

2. **Backend:**
    LangChain: Text processing, embedding generation, and conversational chain   
    setup.
    PyPDF2: Extract text from PDF files.
    FAISS: Efficient similarity search for vector embeddings.
    Google Generative AI: AI models for embedding and conversational tasks.

3. **Utilities:**
    dotenv: Manage environment variables securely.
``````````````````````````````````````````````````````````````````````````````````
# Setup Instructions
**Prerequisites**
    1. Python 3.8 or higher.
    2. API Key for Google Generative AI.
    3. Basic knowledge of Python and Streamlit.
``````````````````````````````````````````````````````````````````````````````````
# Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/pdf-chat-app.git
    cd pdf-chat-app
    ```

2. **Create a virtual environment:**
     ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate     # For Windows
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a .env file in the project root and add your Google API key:
    ```bash
    GOOGLE_API_KEY=your-google-api-key
    ```

5. **Run the application:**
    ```bash
    streamlit run app.py
    ```

6. Open your browser and navigate to the local URL displayed by Streamlit.
``````````````````````````````````````````````````````````````````````````````````
# Project Structure
    ```bash
    📂 pdf-chat-app
    ├── app.py                  # Main application script
    ├── requirements.txt        # Python dependencies
    ├── .env                    # Environment variables file
    ├── README.md               # Project documentation
    └── vector_index/           # Saved FAISS index files
    ```
``````````````````````````````````````````````````````````````````````````````````
# Usage
1. **Upload PDFs:**
    Use the sidebar to upload one or more PDFs.
    Click "Process PDFs" to extract and index the content.

2. **Ask Questions:**
    Enter your question in the text input box.
    The AI will generate a response based on the PDF content.

3. **View Responses:**
    The response is displayed on the main page below the input box.

4. **Future Enhancements**
    Add support for more file types (e.g., Word, Excel).
    Implement caching for processed PDFs to avoid reprocessing.
    Integrate support for multiple language queries.
    Add options for summarizing entire PDFs.
``````````````````````````````````````````````````````````````````````````````````
# Contributing
    Contributions are welcome! If you'd like to contribute, please fork the   
    repository, create a new branch, and submit a pull request. Ensure your code 
    adheres to the project's style and guidelines.
``````````````````````````````````````````````````````````````````````````````````
# License
    This project is licensed under the MIT License. See the LICENSE file for more     details.
``````````````````````````````````````````````````````````````````````````````````
# Acknowledgements
    Streamlit for an easy-to-use web app framework.
    Google Generative AI for powerful AI embeddings and models.
    LangChain for simplifying AI integration.
    FAISS for efficient similarity search.
``````````````````````````````````````````````````````````````````````````````````
