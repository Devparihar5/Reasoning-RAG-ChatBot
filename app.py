import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import logging
import PyPDF2
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Reasoning Enhanced RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4257B2;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5C7AEA;
        margin-bottom: 0.5rem;
    }
    .document-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .document-metadata {
        font-size: 0.8rem;
        color: #555;
        margin-bottom: 5px;
    }
    .reasoning-box {
        background-color: #f9f9f9;
        border-left: 3px solid #5C7AEA;
        padding: 10px;
        margin: 10px 0;
    }
    .chat-message-user {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .chat-message-assistant {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4257B2;
    }
</style>
""", unsafe_allow_html=True)

class PDFProcessor:
    """Process PDF files and extract text content"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> List[Dict[str, str]]:
        """Extract text from PDF file and split into chunks with metadata"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        chunks_with_metadata = []
        
        # Process each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            if text.strip():  # Only add non-empty text
                # Split long pages into smaller chunks (around 500 characters each)
                if len(text) > 500:
                    words = text.split()
                    current_chunk = ""
                    chunk_num = 1
                    
                    for word in words:
                        if len(current_chunk) + len(word) + 1 < 500:
                            if current_chunk:
                                current_chunk += " " + word
                            else:
                                current_chunk = word
                        else:
                            # Add chunk with metadata
                            chunks_with_metadata.append({
                                "text": current_chunk,
                                "metadata": {
                                    "filename": pdf_file.name,
                                    "page": page_num + 1,
                                    "chunk": chunk_num
                                }
                            })
                            current_chunk = word
                            chunk_num += 1
                            
                    if current_chunk:  # Add the last chunk
                        chunks_with_metadata.append({
                            "text": current_chunk,
                            "metadata": {
                                "filename": pdf_file.name,
                                "page": page_num + 1,
                                "chunk": chunk_num
                            }
                        })
                else:
                    # Add the whole page as one chunk
                    chunks_with_metadata.append({
                        "text": text,
                        "metadata": {
                            "filename": pdf_file.name,
                            "page": page_num + 1,
                            "chunk": 1
                        }
                    })
        
        return chunks_with_metadata


class DocumentStore:
    """Store and retrieve documents with vector embeddings"""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the document store with an embedding model"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.document_embeddings = None
        self.index = None

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the store and update index"""
        start_idx = len(self.documents)
        
        # Add ID to each document if not present
        for i, doc in enumerate(documents):
            if "id" not in doc:
                doc["id"] = start_idx + i
        
        self.documents.extend(documents)

        # Extract text for embedding
        texts = [doc["text"] for doc in documents]

        # Generate embeddings
        with st.spinner('Generating document embeddings...'):
            new_embeddings = self.embedding_model.encode(texts)

        if self.document_embeddings is None:
            self.document_embeddings = new_embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])

        # Build or update FAISS index
        self._build_index()
        
        return len(documents)

    def _build_index(self):
        """Build FAISS index for fast similarity search"""
        if self.document_embeddings is None or len(self.document_embeddings) == 0:
            st.warning("No documents to index")
            return
            
        vector_dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        self.index.add(self.document_embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to the query"""
        if self.index is None or len(self.documents) == 0:
            st.warning("No documents in the index to search")
            return []
            
        query_embedding = self.embedding_model.encode([query])

        # Search the index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))

        # Return the top k documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(distances[0][i])
                results.append(doc)

        return results

    def clear(self):
        """Clear all documents and reset the index"""
        self.documents = []
        self.document_embeddings = None
        self.index = None


class ReasoningModule:
    """Module for generating chain-of-thought reasoning"""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize with a reasoning model"""
        with st.spinner('Loading reasoning model...'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_reasoning(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate reasoning steps for a query given context"""
        # Prepare reasoning prompt
        prompt = self._create_reasoning_prompt(query, context)

        # Generate reasoning
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_beams=3,
            early_stopping=True
        )

        reasoning = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reasoning

    def _create_reasoning_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Create a prompt for the reasoning model"""
        # Include metadata in the context
        context_entries = []
        for i, doc in enumerate(context):
            metadata = doc.get("metadata", {})
            metadata_str = ""
            if metadata:
                file_info = f"File: {metadata.get('filename', 'Unknown')}"
                page_info = f"Page: {metadata.get('page', 'Unknown')}"
                metadata_str = f"[{file_info}, {page_info}]"
                
            context_entries.append(f"Document {i+1} {metadata_str}: {doc['text']}")
            
        context_str = "\n\n".join(context_entries)

        prompt = f"""
Given the following context information and question, reason step by step to find the answer.

Context:
{context_str}

Question: {query}

Let's think about this step by step:
"""
        return prompt


class RAGReasoner:
    """Main class that combines retrieval and reasoning"""

    def __init__(
        self,
        document_store: DocumentStore,
        reasoning_module: ReasoningModule,
        model_name: str = "google/flan-t5-base",
        retrieval_k: int = 5
    ):
        """Initialize with components and parameters"""
        self.document_store = document_store
        self.reasoning_module = reasoning_module
        with st.spinner('Loading language model...'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.retrieval_k = retrieval_k  # number of documents to retrieve during the search process

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the entire pipeline"""
        # Check if document store has documents
        if not self.document_store.documents:
            return {
                "query": query,
                "refined_query": query,
                "reasoning": "No documents available to reason with.",
                "initial_docs": [],
                "refined_docs": [],
                "answer": "I don't have any documents to search through. Please upload some PDF files first."
            }
            
        # 1. Initial document retrieval
        with st.spinner('Retrieving relevant documents...'):
            initial_docs = self.document_store.search(query, self.retrieval_k)

        # 2. Generate reasoning based on retrieved documents
        with st.spinner('Generating reasoning...'):
            reasoning = self.reasoning_module.generate_reasoning(query, initial_docs)

        # 3. Use reasoning to refine the query
        with st.spinner('Refining query...'):
            refined_query = self._refine_query(query, reasoning)

        # 4. Retrieve documents again with the refined query
        with st.spinner('Retrieving documents with refined query...'):
            refined_docs = self.document_store.search(refined_query, self.retrieval_k)

        # 5. Generate the final answer
        with st.spinner('Generating final answer...'):
            answer = self._generate_answer(query, reasoning, refined_docs)

        return {
            "query": query,
            "refined_query": refined_query,
            "reasoning": reasoning,
            "initial_docs": initial_docs,
            "refined_docs": refined_docs,
            "answer": answer
        }

    def _refine_query(self, original_query: str, reasoning: str) -> str:
        """Refine the query based on reasoning"""
        prompt = f"""
Original query: {original_query}

Reasoning process:
{reasoning}

Based on this reasoning, provide a refined and expanded search query that would better retrieve relevant information:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=128,
            num_beams=3,
            early_stopping=True
        )

        refined_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return refined_query

    def _generate_answer(self, query: str, reasoning: str, documents: List[Dict[str, Any]]) -> str:
        """Generate final answer based on query, reasoning and documents"""
        # Create context string from documents with metadata
        context_entries = []
        for i, doc in enumerate(documents):
            metadata = doc.get("metadata", {})
            metadata_str = ""
            if metadata:
                file_info = f"File: {metadata.get('filename', 'Unknown')}"
                page_info = f"Page: {metadata.get('page', 'Unknown')}"
                metadata_str = f"[{file_info}, {page_info}]"
                
            context_entries.append(f"Document {i+1} {metadata_str}: {doc['text']}")
            
        context_str = "\n\n".join(context_entries)

        prompt = f"""
Query: {query}

Reasoning process:
{reasoning}

Retrieved documents:
{context_str}

Based on the reasoning and documents, provide a comprehensive and accurate answer to the query:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


class ReasoningRAGChatbot:
    """Chatbot interface for the RAG Reasoning system"""

    def __init__(self, rag_reasoner: RAGReasoner):
        """Initialize with RAGReasoner"""
        self.rag_reasoner = rag_reasoner
        self.conversation_history = []

    def chat(self, user_input: str) -> str:
        """Process user input and generate a response"""
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Process the query
        result = self.rag_reasoner.process_query(user_input)

        # Format a response message
        response = f"{result['answer']}"

        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response, result

    def get_detailed_response(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return detailed results including reasoning"""
        self.conversation_history.append({"role": "user", "content": user_input})

        result = self.rag_reasoner.process_query(user_input)

        response = f"{result['answer']}"
        self.conversation_history.append({"role": "assistant", "content": response})

        return result

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class Dataset:
    """Utility class for loading datasets"""

    @staticmethod
    def create_documents_from_chunks(chunks_with_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create document objects from text chunks with metadata"""
        documents = []

        for i, chunk_data in enumerate(chunks_with_metadata):
            doc = {
                "id": i,
                "text": chunk_data["text"],
                "metadata": chunk_data["metadata"]
            }
            documents.append(doc)

        return documents


# Streamlit UI
def main():
    # Title with custom styling
    st.markdown('<h1 class="main-header">🧠 Reasoning Enhanced RAG Chatbot</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This chatbot combines retrieval-augmented generation with chain-of-thought reasoning to provide more accurate answers.
    Upload PDF documents and ask questions about their content.
    """)

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    if 'document_store' not in st.session_state:
        st.session_state.document_store = DocumentStore()
    
    if 'reasoning_module' not in st.session_state:
        with st.spinner("Initializing reasoning module..."):
            st.session_state.reasoning_module = ReasoningModule()
    
    if 'rag_reasoner' not in st.session_state:
        with st.spinner("Initializing RAG reasoner..."):
            st.session_state.rag_reasoner = RAGReasoner(
                st.session_state.document_store, 
                st.session_state.reasoning_module
            )
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ReasoningRAGChatbot(st.session_state.rag_reasoner)
    
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        
    if 'document_count' not in st.session_state:
        st.session_state.document_count = 0
        
    if 'show_details' not in st.session_state:
        st.session_state.show_details = True

    # Sidebar for settings and document upload
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📄 Document Management</h2>', unsafe_allow_html=True)
        
        # File uploader with better styling
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload one or more PDF files to ask questions about"
        )
        
        if uploaded_files:
            # Check if new files were uploaded
            current_file_names = [file.name for file in uploaded_files]
            previous_file_names = [file.name for file in st.session_state.uploaded_files] if st.session_state.uploaded_files else []
            
            if current_file_names != previous_file_names:
                st.session_state.uploaded_files = uploaded_files
                st.session_state.documents_loaded = False
        
        # Process and clear buttons in a row
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("📥 Process Files", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear All", use_container_width=True)
            
        if process_button and uploaded_files:
            with st.spinner("Processing PDF documents..."):
                # Clear existing documents
                st.session_state.document_store.clear()
                st.session_state.document_count = 0
                
                all_chunks_with_metadata = []
                
                # Process each PDF file
                progress_bar = st.progress(0)
                for i, pdf_file in enumerate(uploaded_files):
                    # Extract text from PDF with metadata
                    chunks_with_metadata = st.session_state.pdf_processor.extract_text_from_pdf(pdf_file)
                    all_chunks_with_metadata.extend(chunks_with_metadata)
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                
                # Create documents from text chunks
                documents = Dataset.create_documents_from_chunks(all_chunks_with_metadata)
                
                # Add documents to document store
                doc_count = st.session_state.document_store.add_documents(documents)
                st.session_state.document_count = doc_count
                
                st.session_state.documents_loaded = True
                st.success(f"✅ Processed {len(uploaded_files)} PDF files with {doc_count} text chunks")
                
                # Clear conversation history when new documents are loaded
                st.session_state.conversation = []
                st.session_state.chatbot.clear_history()
                
                # Small delay to ensure the success message is seen
                time.sleep(1)
                
        if clear_button:
            st.session_state.document_store.clear()
            st.session_state.documents_loaded = False
            st.session_state.document_count = 0
            st.session_state.conversation = []
            st.session_state.chatbot.clear_history()
            st.success("🧹 All documents and conversation history cleared!")
        
        # Document stats with metrics
        if st.session_state.document_count > 0:
            st.markdown("### 📊 Document Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files", len(st.session_state.uploaded_files))
            with col2:
                st.metric("Text Chunks", st.session_state.document_count)
        
        # Settings section
        st.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)
        
        # Toggle for showing reasoning details
        st.session_state.show_details = st.toggle("Show reasoning details", value=st.session_state.show_details)
        
        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.chatbot.clear_history()
            st.success("Conversation cleared!")
        
        # About section
        st.markdown('<h2 class="sub-header">ℹ️ About</h2>', unsafe_allow_html=True)
        st.markdown("""
        This chatbot uses a Retrieval-Augmented Generation (RAG) system enhanced with chain-of-thought reasoning.
        
        **Components:**
        - 🔍 Document Store with FAISS vector search
        - 🧠 Reasoning Module using Flan-T5-Base
        - 🔄 Query refinement based on reasoning
        - 💬 Final answer generation
        """)

    # Main content area - Tabs for Chat and Document Explorer
    if not st.session_state.documents_loaded:
        st.info("👆 Please upload PDF documents and click 'Process Files' in the sidebar to start.")
        
        # Show sample questions that can be asked
        st.markdown("### Sample Questions You Can Ask")
        st.markdown("""
        Once you upload documents, you can ask questions like:
        - "What are the main components of the RAG system?"
        - "How does the reasoning module work?"
        - "Explain the document retrieval process"
        - "What is chain-of-thought reasoning?"
        """)
    else:
        # Create tabs for Chat and Document Explorer
        tab1, tab2 = st.tabs(["💬 Chat", "🔍 Document Explorer"])
        
        # Chat Tab
        with tab1:
            # Split into chat and details columns if details are enabled
            if st.session_state.show_details:
                chat_col, details_col = st.columns([3, 2])
            else:
                chat_col = st
            
            with chat_col:
                # Chat container for messages
                chat_container = st.container(height=500)
                
                # Display conversation in the container
                with chat_container:
                    for message in st.session_state.conversation:
                        if message["role"] == "user":
                            with st.chat_message("user", avatar="👤"):
                                st.write(message["content"])
                        else:
                            with st.chat_message("assistant", avatar="🤖"):
                                st.write(message["content"])
            
            # Details column for showing reasoning and documents
            if st.session_state.show_details and len(st.session_state.conversation) > 0:
                with details_col:
                    # Find the last assistant message with details
                    details = None
                    for message in reversed(st.session_state.conversation):
                        if message["role"] == "assistant" and "details" in message:
                            details = message["details"]
                            break
                    
                    if details:
                        st.markdown('<h3 class="sub-header">Reasoning Process</h3>', unsafe_allow_html=True)
                        
                        # Initial and refined query
                        with st.expander("Query Analysis", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Initial Query:**")
                                st.info(details["query"])
                            with col2:
                                st.markdown("**Refined Query:**")
                                st.success(details["refined_query"])
                        
                        # Step-by-Step Reasoning
                        with st.expander("Step-by-Step Reasoning", expanded=True):
                            st.markdown('<div class="reasoning-box">' + details["reasoning"] + '</div>', unsafe_allow_html=True)
                        
                        # Retrieved Documents
                        st.markdown('<h3 class="sub-header">Retrieved Documents</h3>', unsafe_allow_html=True)
                        
                        # Show documents in a more visual way
                        for i, doc in enumerate(details["refined_docs"][:3]):  # Limit to top 3 for space
                            with st.expander(f"Document {i+1} (Score: {doc['score']:.4f})", expanded=i==0):
                                metadata = doc.get("metadata", {})
                                if metadata:
                                    st.markdown(f"**Source:** {metadata.get('filename', 'Unknown')} | **Page:** {metadata.get('page', 'Unknown')}")
                                st.markdown('<div class="document-card">' + doc["text"] + '</div>', unsafe_allow_html=True)
            
            # Chat input at the bottom
            user_input = st.chat_input("Ask a question about your documents...")
            
            # Process user input
            if user_input:
                # Add user message to conversation
                st.session_state.conversation.append({"role": "user", "content": user_input})
                
                # Get chatbot response
                response, details = st.session_state.chatbot.chat(user_input)
                
                # Add assistant response to conversation
                st.session_state.conversation.append({
                    "role": "assistant", 
                    "content": response,
                    "details": details
                })
                
                # Force a rerun to update the UI
                st.rerun()
        
        # Document Explorer Tab
        with tab2:
            st.markdown('<h3 class="sub-header">Document Explorer</h3>', unsafe_allow_html=True)
            
            # Group documents by filename
            files = {}
            for doc in st.session_state.document_store.documents:
                filename = doc.get("metadata", {}).get("filename", "Unknown")
                if filename not in files:
                    files[filename] = []
                files[filename].append(doc)
            
            # Display documents by file
            for filename, docs in files.items():
                with st.expander(f"📄 {filename} ({len(docs)} chunks)"):
                    # Group by page
                    pages = {}
                    for doc in docs:
                        page = doc.get("metadata", {}).get("page", "Unknown")
                        if page not in pages:
                            pages[page] = []
                        pages[page].append(doc)
                    
                    # Display by page
                    for page, page_docs in sorted(pages.items()):
                        st.markdown(f"**Page {page}**")
                        for i, doc in enumerate(page_docs):
                            st.markdown(f"<div class='document-card'><div class='document-metadata'>Chunk {doc.get('metadata', {}).get('chunk', i+1)}</div>{doc['text']}</div>", unsafe_allow_html=True)
                        st.markdown("---")

if __name__ == "__main__":
    main()
