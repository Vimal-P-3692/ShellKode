"""
Vector Store Module
Handles document chunking, embeddings, and vector store management.
"""
import os
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """Class to manage document chunking and vector store operations."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "models/embedding-001"
    ):
        """
        Initialize the vector store manager.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of the Google embedding model
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS vector store
        """
        # Split documents into chunks
        chunks = self.create_chunks(documents)
        
        # Create vector store
        print("Creating embeddings and building vector store...")
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        print("Vector store created successfully!")
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to an existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create it first.")
        
        chunks = self.create_chunks(documents)
        print(f"Adding {len(chunks)} new chunks to vector store...")
        self.vector_store.add_documents(chunks)
        print("Documents added successfully!")
    
    def save_vector_store(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        os.makedirs(path, exist_ok=True)
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str) -> FAISS:
        """
        Load a vector store from disk.
        
        Args:
            path: Directory path containing the saved vector store
            
        Returns:
            FAISS vector store
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        print(f"Loading vector store from {path}...")
        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully!")
        return self.vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever instance for use with LangChain agents.
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            Retriever object
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
