"""
PDF Processor Module
Handles loading and extracting text from PDF documents.
"""
import os
from typing import List
from PyPDF2 import PdfReader
from langchain_core.documents import Document


class PDFProcessor:
    """Class to handle PDF loading and text extraction."""
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_path: Path to the PDF file
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.reader = None
        self.documents = []
    
    def load_pdf(self) -> List[Document]:
        """
        Load and extract text from the PDF file.
        
        Returns:
            List of Document objects containing the extracted text
        """
        try:
            self.reader = PdfReader(self.pdf_path)
            print(f"Loading PDF: {self.pdf_path}")
            print(f"Total pages: {len(self.reader.pages)}")
            
            documents = []
            for page_num, page in enumerate(self.reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": self.pdf_path,
                            "page": page_num + 1,
                            "total_pages": len(self.reader.pages)
                        }
                    )
                    documents.append(doc)
            
            self.documents = documents
            print(f"Successfully extracted text from {len(documents)} pages")
            return documents
        
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def get_metadata(self) -> dict:
        """
        Get metadata information from the PDF.
        
        Returns:
            Dictionary containing PDF metadata
        """
        if not self.reader:
            self.load_pdf()
        
        metadata = {
            "file_path": self.pdf_path,
            "file_name": os.path.basename(self.pdf_path),
            "total_pages": len(self.reader.pages),
            "pdf_metadata": self.reader.metadata if self.reader.metadata else {}
        }
        
        return metadata


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """
    Convenience function to load PDF documents.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects
    """
    processor = PDFProcessor(pdf_path)
    return processor.load_pdf()
