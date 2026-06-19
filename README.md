# PDF Chat Application

Interact with your PDF files in a conversational way! This Streamlit-based application allows you to upload multiple PDFs, process their content, and ask questions about the text extracted from the documents. Using advanced AI models from Google Generative AI and Hugging Face, this tool provides detailed, accurate, and context-aware responses.

## Features

1. **PDF Upload and Processing:**
   * Upload multiple PDF files.
   * Extract text from PDFs efficiently using `PyPDF2`.
2. **Text Chunking:**
   * Split large text into manageable chunks for better context handling using LangChain's `RecursiveCharacterTextSplitter`.
3. **Vector Embedding and Search:**
   * Generate vector embeddings for text chunks using Hugging Face `all-MiniLM-L6-v2`.
   * Store and index embeddings locally using a `FAISS` vector store for efficient similarity-based context retrieval.
4. **AI-Powered Q&A:**
   * Query the index and generate responses using Google Generative AI's `gemini-flash-latest` model.
5. **Streamlit Interface:**
   * User-friendly multi-tab interface (Upload & About) with customized styling.

---

## Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Interactive web framework)
* **LLM Engine:** [Google Generative AI](https://ai.google.dev/) (`gemini-flash-latest`)
* **Embeddings:** Hugging Face sentence-transformers (`all-MiniLM-L6-v2`) via `langchain-community`
* **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
* **Text Splitting/Orchestration:** [LangChain](https://www.langchain.com/)
* **PDF Parser:** `PyPDF2`
* **Utilities:** `python-dotenv` (secure configuration management)

---

## Project Structure

```text
📂 PDF_chat_application
├── .streamlit/
│   └── config.toml          # Streamlit server config (e.g. maxUploadSize)
├── app.py                   # Main Streamlit application script
├── requirements.txt         # Python package dependencies
├── LICENSE                  # MIT License file
├── README.md                # Project documentation
├── .gitignore               # Git ignore rules
├── .env                     # Local environment variables (ignored by Git)
└── vector_index/            # Saved FAISS index files (generated locally, ignored by Git)
```

---

## Setup Instructions

### Prerequisites
* Python 3.8 or higher.
* A Google AI Studio API Key (for Gemini access).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/piyush06singhal/PDF_chat_application.git
   cd PDF_chat_application
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   * **On Windows:**
     ```bash
     venv\Scripts\activate
     ```
   * **On macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   Create a `.env` file in the project root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```

7. Open your browser and navigate to the local URL displayed by Streamlit (usually `http://localhost:8501`).

---

## Usage

1. **Upload PDFs:**
   * Go to the **Upload PDFs** tab.
   * Drag and drop one or more PDF files.
   * Click the **Process PDFs** button to extract text, generate embeddings, and build the local FAISS vector index.
2. **Ask Questions:**
   * Once processing completes, a question input box will appear.
   * Type your query related to the uploaded PDF content.
   * The AI will retrieve the most relevant sections of the document and provide a context-informed response.

---

## Future Enhancements
* Add support for more file types (e.g., Docx, Txt, Markdown).
* Implement session state memory to support multi-turn conversation/chat history.
* Integrate support for multilingual queries.
* Add option to summarize entire PDFs at once.

---

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository, create a new branch, and submit a pull request. Ensure your code adheres to the project's style and guidelines.

## License
This project is licensed under the MIT License. See the [LICENSE](file:///C:/Users/Piyush/OneDrive/Desktop/PDF_chat_application/LICENSE) file for more details.

## Acknowledgements
* Streamlit for an easy-to-use web app framework.
* Google Generative AI for powerful AI language models.
* LangChain for simplifying AI integration and processing.
* FAISS for efficient similarity search.
