"""
Example usage of the PDF Question Answering Agent
This script demonstrates different ways to use the agent
"""

import os
from pdf_qa_agent import PDFQuestionAnsweringAgent


def example_basic_usage():
    """
    Basic usage: Process a PDF and ask questions
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Initialize the agent
    agent = PDFQuestionAnsweringAgent(
        model="gpt-4",  # Use "gpt-3.5-turbo" for faster/cheaper option
        embedding_model="text-embedding-ada-002"
    )
    
    # Process your PDF
    pdf_path = "sample.pdf"  # Replace with your PDF path
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please place a PDF file at: {pdf_path}")
        return
    
    agent.process_pdf(pdf_path, chunk_size=1000, overlap=200)
    
    # Ask a single question
    question = "What is the main topic of this document?"
    result = agent.answer_question(question)
    
    print(f"\n📝 Question: {result['question']}")
    print(f"✨ Answer: {result['answer']}")


def example_multiple_questions():
    """
    Ask multiple questions about the same PDF
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Questions")
    print("="*70)
    
    agent = PDFQuestionAnsweringAgent()
    
    pdf_path = "sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please place a PDF file at: {pdf_path}")
        return
    
    agent.process_pdf(pdf_path)
    
    # Ask multiple questions
    questions = [
        "What is the main topic?",
        "Who are the key people mentioned?",
        "What are the main conclusions?",
        "Are there any important dates mentioned?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"Question {i}: {question}")
        print('─'*70)
        
        result = agent.answer_question(question, top_k=3)
        print(f"Answer: {result['answer']}")


def example_save_and_load_embeddings():
    """
    Save embeddings for reuse (avoid reprocessing the same PDF)
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Save and Load Embeddings")
    print("="*70)
    
    agent = PDFQuestionAnsweringAgent()
    
    pdf_path = "sample.pdf"
    embeddings_path = "embeddings.pkl"
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please place a PDF file at: {pdf_path}")
        return
    
    # Check if embeddings already exist
    if os.path.exists(embeddings_path):
        print("\n✓ Loading existing embeddings...")
        agent.load_embeddings(embeddings_path)
    else:
        print("\n✓ Processing PDF and creating embeddings...")
        agent.process_pdf(pdf_path)
        agent.save_embeddings(embeddings_path)
    
    # Now you can ask questions
    result = agent.answer_question("Summarize the key points")
    print(f"\n✨ Answer: {result['answer']}")


def example_with_context_display():
    """
    Show the relevant context chunks along with the answer
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Display Relevant Context")
    print("="*70)
    
    agent = PDFQuestionAnsweringAgent()
    
    pdf_path = "sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please place a PDF file at: {pdf_path}")
        return
    
    agent.process_pdf(pdf_path)
    
    question = "What are the main findings?"
    result = agent.answer_question(question, top_k=3)
    
    print(f"\n📝 Question: {result['question']}")
    print(f"\n✨ Answer:\n{result['answer']}")
    
    print(f"\n📄 Relevant Context Chunks:")
    print("="*70)
    
    for i, (chunk, score) in enumerate(result['relevant_chunks'], 1):
        print(f"\n[Chunk {i}] Similarity Score: {score:.4f}")
        print("─"*70)
        # Show first 300 characters of each chunk
        preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
        print(preview)


def example_interactive_mode():
    """
    Interactive Q&A session
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Interactive Mode")
    print("="*70)
    
    agent = PDFQuestionAnsweringAgent()
    
    pdf_path = "sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please place a PDF file at: {pdf_path}")
        print("Creating a sample PDF path for demonstration...")
        return
    
    agent.process_pdf(pdf_path)
    
    # Start interactive mode
    agent.interactive_mode()


def example_with_env_file():
    """
    Use environment variables for API key
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Using .env file for API key")
    print("="*70)
    
    # Create a .env file with:
    # OPENAI_API_KEY=your-api-key-here
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # The agent will automatically read from OPENAI_API_KEY env variable
    agent = PDFQuestionAnsweringAgent()
    
    pdf_path = "sample.pdf"
    
    if os.path.exists(pdf_path):
        agent.process_pdf(pdf_path)
        result = agent.answer_question("What is this document about?")
        print(f"\n✨ Answer: {result['answer']}")
    else:
        print(f"⚠️  Please place a PDF file at: {pdf_path}")


def example_custom_parameters():
    """
    Customize chunk size, overlap, and number of relevant chunks
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Custom Parameters")
    print("="*70)
    
    agent = PDFQuestionAnsweringAgent(
        model="gpt-3.5-turbo",  # Faster and cheaper
        embedding_model="text-embedding-ada-002"
    )
    
    pdf_path = "sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please place a PDF file at: {pdf_path}")
        return
    
    # Custom chunking parameters
    agent.process_pdf(
        pdf_path,
        chunk_size=1500,  # Larger chunks
        overlap=300       # More overlap
    )
    
    # Ask question with custom parameters
    result = agent.answer_question(
        "What are the key points?",
        top_k=7,          # Consider more chunks
        temperature=0.7   # More creative responses
    )
    
    print(f"\n✨ Answer: {result['answer']}")


def main():
    """
    Run all examples (comment out the ones you don't want to run)
    """
    print("\n" + "="*70)
    print("PDF QUESTION ANSWERING AGENT - EXAMPLES")
    print("="*70)
    print("\nNote: Make sure to:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Place a PDF file named 'sample.pdf' in the current directory")
    print("3. Install requirements: pip install -r requirements.txt")
    
    # Uncomment the example you want to run:
    
    # example_basic_usage()
    # example_multiple_questions()
    # example_save_and_load_embeddings()
    # example_with_context_display()
    # example_interactive_mode()
    # example_with_env_file()
    # example_custom_parameters()
    
    print("\n" + "="*70)
    print("💡 Tip: Uncomment the examples you want to run in the main() function")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
