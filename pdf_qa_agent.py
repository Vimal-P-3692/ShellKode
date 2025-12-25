"""
CLI Interface for PDF Question Answering Agent
Command-line interface for interacting with the PDF QA system.
"""
import os
import sys
import glob
from dotenv import load_dotenv
from pdf_processor import load_pdf_documents
from vector_store import VectorStoreManager
from qa_agent import PDFQAAgent

# Load environment variables
load_dotenv()


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   📄 AI PDF Question Answering Agent                 ║
║   Powered by LangChain & Google Gemini                 ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def print_separator():
    """Print a separator line."""
    print("\n" + "─" * 60 + "\n")


def find_pdf_in_folder():
    """
    Find the first PDF file in the current directory.
    
    Returns:
        Path to the PDF file or None if not found
    """
    pdf_files = glob.glob("*.pdf")
    if pdf_files:
        return pdf_files[0]
    return None
    print("\n" + "─" * 60 + "\n")


def initialize_system(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Initialize the PDF QA system.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Tuple of (vector_store_manager, qa_agent)
    """
    try:
        print(f"\n🔄 Loading PDF: {pdf_path}")
        print_separator()
        
        # Load PDF documents
        documents = load_pdf_documents(pdf_path)
        print(f"✅ Successfully loaded {len(documents)} pages\n")
        
        # Create vector store
        print("🔄 Creating embeddings and vector store...")
        print("   This may take a moment...\n")
        vector_store_manager = VectorStoreManager(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        vector_store = vector_store_manager.create_vector_store(documents)
        print("✅ Vector store created successfully!\n")
        
        # Initialize QA agent
        print("🔄 Initializing QA Agent...")
        retriever = vector_store_manager.get_retriever(search_kwargs={"k": 4})
        qa_agent = PDFQAAgent(
            retriever=retriever,
            model_name="gemini-2.0-flash",
            temperature=0.0
        )
        print("✅ QA Agent ready!\n")
        
        print_separator()
        return vector_store_manager, qa_agent
    
    except Exception as e:
        print(f"\n❌ Error initializing system: {str(e)}")
        sys.exit(1)


def display_response(response: dict):
    """
    Display the agent's response.
    
    Args:
        response: Response dictionary from the agent
    """
    print("\n🤖 Answer:")
    print_separator()
    print(response["answer"])
    
    # Display sources
    if response["sources"]:
        print_separator()
        print("📚 Sources:")
        for idx, source in enumerate(response["sources"], 1):
            print(f"\n  [{idx}] Page {source['page']}:")
            print(f"      {source['content'][:150]}...")
    
    print_separator()


def interactive_mode(qa_agent: PDFQAAgent):
    """
    Run interactive question-answering mode.
    
    Args:
        qa_agent: Initialized QA agent
    """
    print("\n✨ Ready to answer your questions!")
    print("   Type 'quit' or 'exit' to end the session")
    print("   Type 'clear' to clear conversation memory")
    print_separator()
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Your Question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the PDF QA Agent. Goodbye!\n")
                break
            
            # Check for clear command
            if question.lower() == 'clear':
                qa_agent.clear_memory()
                print("✅ Conversation memory cleared!")
                continue
            
            # Skip empty questions
            if not question:
                continue
            
            # Get answer from agent
            print("\n🔄 Processing your question...")
            response = qa_agent.ask(question)
            
            # Display response
            display_response(response)
        
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using the PDF QA Agent. Goodbye!\n")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def main():
    """Main CLI function."""
    print_banner()
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("   Please create a .env file with your Google API key")
        sys.exit(1)
    
    # Get PDF path from command line or auto-detect
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Try to find a PDF in the current folder
        pdf_path = find_pdf_in_folder()
        if pdf_path:
            print(f"📄 Found PDF: {pdf_path}")
        else:
            print("❌ Error: No PDF file found in the current directory")
            print("   Please place a PDF file in this folder or specify the path:")
            print("   python pdf_qa_agent.py <path_to_pdf>")
            sys.exit(1)
    
    # Validate PDF path
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.lower().endswith('.pdf'):
        print("❌ Error: File must be a PDF")
        sys.exit(1)
    
    # Initialize system
    vector_store_manager, qa_agent = initialize_system(pdf_path)
    
    # Run interactive mode
    interactive_mode(qa_agent)


if __name__ == "__main__":
    main()
