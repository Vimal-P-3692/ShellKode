"""
Quick Start Script - Simple example to get you started quickly
"""

from pdf_qa_agent import PDFQuestionAnsweringAgent
import os


def main():
    print("="*70)
    print("PDF Question Answering Agent - Quick Start")
    print("="*70)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  ERROR: GEMINI_API_KEY environment variable not set!")
        print("\nPlease set it using:")
        print('  export GEMINI_API_KEY="your-api-key-here"')
        print("\nOr create a .env file with:")
        print('  GEMINI_API_KEY=your-api-key-here')
        return
    
    # Get PDF path from user
    pdf_path = input("\n📄 Enter the path to your PDF file: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"\n❌ Error: File not found: {pdf_path}")
        return
    
    print("\n" + "="*70)
    print("Initializing Agent...")
    print("="*70)
    
    # Initialize agent
    agent = PDFQuestionAnsweringAgent(model="gemini-2.5-flash")
    
    # Process PDF
    agent.process_pdf(pdf_path)
    
    print("\n" + "="*70)
    print("PDF Processed Successfully! You can now ask questions.")
    print("="*70)
    
    # Start interactive mode
    agent.interactive_mode()


if __name__ == "__main__":
    main()
