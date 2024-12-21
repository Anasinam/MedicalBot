import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Directory containing PDFs
PDF_FOLDER = "C:/Chatbot/MediGuide/PDFs"  # Path to your PDF files
VECTOR_STORE_PATH = "faiss_index"  # Path to save the FAISS vector store

# Function to extract text from PDFs
def get_pdf_text(pdf_folder):
    text = ""
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return text

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Function to create and save the vector store
def create_vector_store(pdf_folder, vector_store_path):
    # Extract and chunk text from PDFs
    raw_text = get_pdf_text(pdf_folder)
    if not raw_text.strip():
        print("No text extracted from the PDFs.")
        return

    text_chunks = get_text_chunks(raw_text)

    # Generate embeddings and create FAISS vector store
    print("Generating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the vector store
    print(f"Saving vector store to {vector_store_path}...")
    vector_store.save_local(vector_store_path)
    print("Training complete. Vector store saved.")

if __name__ == "__main__":
    # Run the training process
    create_vector_store(PDF_FOLDER, VECTOR_STORE_PATH)
