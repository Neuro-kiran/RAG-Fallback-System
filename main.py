import streamlit as st
import logging
from pathlib import Path
from typing import List
import json
import os
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from the .env file
load_dotenv()

# LangChain and HuggingFace imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DocumentProcessor Class
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_file(self, file_path: str) -> List[str]:
        try:
            file_extension = Path(file_path).suffix.lower()

            # Load PDF or TXT file
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                text_content = [page.page_content for page in pages]
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
                documents = loader.load()
                text_content = [doc.page_content for doc in documents]
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Split text into chunks
            chunks = []
            for text in text_content:
                chunks.extend(self.text_splitter.split_text(text))
            return chunks

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

# Enhanced Retrieval-Augmented Generation (RAG) Class
class EnhancedRAG:
    def __init__(self):
        # Use the API key from the environment variables
        api_key = os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("GROQ_API_KEY is not set in the environment variables.")
        os.environ["GROQ_API_KEY"] = api_key
        
        self.llm = ChatGroq(
            model="llama-3.2-11b-text-preview",
            temperature=0.7,
            max_tokens=None
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        self.doc_processor = DocumentProcessor()

        # Define unified prompt template
        self.unified_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a versatile AI assistant that can engage in natural conversation while providing accurate information from both general knowledge and specific documents.

            For each query, follow these steps:
            1. Analyze if the query is a casual interaction, general knowledge, or document-specific question.
            2. Respond warmly to casual interactions.
            3. Provide accurate general knowledge responses if confident, otherwise check documents.
            4. For document-specific queries, use the provided context to answer accurately.
            - Current context (if available): {context}
            """),
            ("human", "{input}")
        ])

    def load_document(self, file_path: str) -> None:
        try:
            # Process the document and create vector store
            chunks = self.doc_processor.process_file(file_path)
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(chunks, self.embeddings)
            else:
                self.vector_store.add_texts(chunks)
            logger.info("Document loaded and vector store updated successfully.")
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise

    def get_response(self, question: str) -> str:
        try:
            # Retrieve relevant context
            context = ""
            if self.vector_store is not None:
                retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                context_docs = retriever.get_relevant_documents(question)
                context = "\n\n".join(doc.page_content for doc in context_docs)

            # Invoke LLM with the unified prompt
            chain = self.unified_prompt | self.llm
            response = chain.invoke({
                "input": question,
                "context": context if context else "No document context available."
            })
            return response.content

        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return "I apologize, but I'm having trouble processing your question. Could you please rephrase it?"

# Streamlit UI
st.title("Enhanced Retrieval-Augmented Generation (RAG) Assistant")

# Create an instance of EnhancedRAG
rag_system = EnhancedRAG()

# Document Upload Section
uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])
if uploaded_file:
    temp_file_path = Path(f"./temp_{uploaded_file.name}")
    temp_file_path.write_bytes(uploaded_file.read())

    # Load the document
    with st.spinner("Processing document..."):
        try:
            rag_system.load_document(str(temp_file_path))
            st.success("Document loaded successfully!")
        except Exception as e:
            st.error(f"Error loading document: {e}")

# User Question Input
question = st.text_input("Ask a question")
if question:
    with st.spinner("Generating response..."):
        response = rag_system.get_response(question)
        st.write("**Assistant:**", response)
