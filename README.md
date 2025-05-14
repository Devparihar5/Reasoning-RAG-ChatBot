# Reasoning Enhanced RAG Chatbot

A chatbot that combines retrieval-augmented generation with chain-of-thought reasoning to provide more accurate and explainable answers.

## Features

- Document retrieval using FAISS vector search
- Chain-of-thought reasoning with Flan-T5-Base
- Query refinement based on reasoning
- Interactive Streamlit interface
- PDF document processing and chunking
- Document explorer with metadata visualization
- Detailed reasoning visualization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Devparihar5/Reasoning-RAG-ChatBot.git
cd Reasoning-RAG-ChatBot
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

Then:
1. Upload PDF documents using the sidebar
2. Click "Process Files" to index the documents
3. Ask questions in the chat interface
4. Toggle "Show reasoning details" to see the reasoning process

## How It Works

1. **PDF Processing**: Uploaded PDFs are processed and split into manageable chunks with metadata.

2. **Initial Document Retrieval**: When a user asks a question, the system retrieves relevant document chunks using vector similarity search.

3. **Reasoning Generation**: The system generates step-by-step reasoning about the retrieved documents using Flan-T5-Base.

4. **Query Refinement**: Based on the reasoning, the original query is refined to better capture the information need.

5. **Final Document Retrieval**: Documents are retrieved again using the refined query.

6. **Answer Generation**: A final answer is generated based on the reasoning and retrieved documents.

## Components

- **PDFProcessor**: Extracts and chunks text from PDF files with metadata
- **DocumentStore**: Manages document embeddings and retrieval using FAISS
- **ReasoningModule**: Generates chain-of-thought reasoning using Flan-T5-Base
- **RAGReasoner**: Orchestrates the entire pipeline
- **ReasoningRAGChatbot**: Provides the chat interface
- **Dataset**: Utility class for managing document collections

## Requirements

- streamlit
- faiss-cpu
- sentence-transformers
- transformers
- torch
- numpy
- PyPDF2
- langchain
- langchain-community

## UI Features

- Document statistics dashboard
- Chat interface with conversation history
- Document explorer to browse uploaded content
- Detailed reasoning visualization
- Query refinement display
- Retrieved document visualization with relevance scores

## Customization

You can modify the following parameters in the code:
- Embedding model (default: all-MiniLM-L6-v2)
- Reasoning model (default: google/flan-t5-base)
- Number of documents to retrieve (default: 5)
- Chunk size for PDF processing (default: ~500 characters)
