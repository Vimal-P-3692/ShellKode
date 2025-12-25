"""
AI PDF Question Answering Agent (without LangChain agents)
A custom implementation using Google Gemini API and TF-IDF for embeddings
"""

import os
import numpy as np
from typing import List, Dict, Tuple
import google.generativeai as genai
from PyPDF2 import PdfReader
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PDFQuestionAnsweringAgent:
    """
    A PDF Question Answering Agent that processes PDFs, creates embeddings,
    and answers questions based on the document content.
    """
    
    def __init__(self, gemini_api_key: str = None, model: str = "gemini-2.5-flash"):
        """
        Initialize the PDF QA Agent
        
        Args:
            gemini_api_key: Google Gemini API key (if None, reads from GEMINI_API_KEY env variable)
            model: Gemini model to use for answering questions
        """
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key must be provided or set in GEMINI_API_KEY environment variable")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.chunks = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.chunk_vectors = None
        self.pdf_path = None
        
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load and extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        self.pdf_path = pdf_path
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        print(f"✓ Loaded PDF: {pdf_path}")
        print(f"✓ Extracted {len(text)} characters from {len(reader.pages)} pages")
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        print(f"✓ Created {len(chunks)} chunks from text")
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create TF-IDF vectors for text chunks (local, no API calls)
        
        Args:
            texts: List of text strings to vectorize
            
        Returns:
            NumPy array of TF-IDF vectors
        """
        print(f"Creating TF-IDF vectors for {len(texts)} chunks...")
        
        # Fit and transform the texts into TF-IDF vectors
        self.chunk_vectors = self.vectorizer.fit_transform(texts)
        
        print("✓ Vectors created successfully (no API quota used!)")
        return self.chunk_vectors
    
    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most relevant chunks for a given query using TF-IDF
        
        Args:
            query: User's question
            top_k: Number of top relevant chunks to return
            
        Returns:
            List of tuples (chunk_text, similarity_score)
        """
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return chunks with their similarity scores
        top_chunks = [(self.chunks[i], similarities[i]) for i in top_indices]
        
        return top_chunks
    
    def answer_question(self, question: str, top_k: int = 5, temperature: float = 0.3) -> Dict:
        """
        Answer a question based on the PDF content
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to consider
            temperature: Gemini temperature parameter (0-1)
            
        Returns:
            Dictionary with answer and relevant context
        """
        if not self.chunks or self.chunk_vectors is None:
            raise ValueError("No PDF has been processed. Please call process_pdf() first.")
        
        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(question, top_k)
        
        # Prepare context from relevant chunks
        context = "\n\n".join([chunk for chunk, score in relevant_chunks])
        
        # Create prompt
        prompt = f"""Based on the following context from a PDF document, please answer the question. 
If the answer cannot be found in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer:"""
        
        # Get answer from Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
            )
        )
        
        answer = response.text
        
        return {
            "question": question,
            "answer": answer,
            "relevant_chunks": relevant_chunks,
            "source": self.pdf_path
        }
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200):
        """
        Complete pipeline: Load PDF, chunk text, and create TF-IDF vectors
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
        """
        print(f"\n{'='*60}")
        print("Processing PDF Document")
        print(f"{'='*60}\n")
        
        # Load PDF
        text = self.load_pdf(pdf_path)
        
        # Chunk text
        self.chunks = self.chunk_text(text, chunk_size, overlap)
        
        # Create TF-IDF vectors (no API calls!)
        self.chunk_vectors = self.create_embeddings(self.chunks)
        
        print(f"\n{'='*60}")
        print("PDF Processing Complete!")
        print(f"{'='*60}\n")
    
    def save_embeddings(self, filepath: str):
        """
        Save chunks and vectors to a file for later use
        
        Args:
            filepath: Path to save the data
        """
        data = {
            "chunks": self.chunks,
            "chunk_vectors": self.chunk_vectors,
            "vectorizer": self.vectorizer,
            "pdf_path": self.pdf_path
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved vectors to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """
        Load previously saved chunks and vectors
        
        Args:
            filepath: Path to the saved data file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data["chunks"]
        self.chunk_vectors = data["chunk_vectors"]
        self.vectorizer = data["vectorizer"]
        self.pdf_path = data["pdf_path"]
        
        print(f"✓ Loaded vectors from {filepath}")
        print(f"✓ Loaded {len(self.chunks)} chunks")
    
    def interactive_mode(self):
        """
        Start an interactive Q&A session
        """
        print("\n" + "="*60)
        print("Interactive PDF Question Answering Mode")
        print("="*60)
        print(f"PDF: {self.pdf_path}")
        print("Type 'exit' or 'quit' to end the session")
        print("="*60 + "\n")
        
        while True:
            question = input("\n🤔 Your Question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                print("\n💭 Thinking...\n")
                result = self.answer_question(question)
                
                print(f"✨ Answer:\n{result['answer']}\n")
                
                # Optionally show relevant chunks
                show_context = input("Show relevant context? (y/n): ").strip().lower()
                if show_context == 'y':
                    print("\n📄 Relevant Context:")
                    for i, (chunk, score) in enumerate(result['relevant_chunks'], 1):
                        print(f"\n[Chunk {i}] (Similarity: {score:.3f})")
                        print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
            
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """
    Example usage of the PDF QA Agent
    """
    # Initialize agent
    agent = PDFQuestionAnsweringAgent(model="gemini-pro")
    
    # Process a PDF
    pdf_path = "your_document.pdf"  # Replace with your PDF path
    agent.process_pdf(pdf_path)
    
    # Save embeddings for future use (optional)
    # agent.save_embeddings("embeddings.pkl")
    
    # Ask questions
    questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
    ]
    
    for question in questions:
        result = agent.answer_question(question)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}\n")
        print("-" * 60)
    
    # Or start interactive mode
    # agent.interactive_mode()


if __name__ == "__main__":
    main()
