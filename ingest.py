import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_faiss_index(pdf_path: str, index_path: str):
    """
    Process a PDF file: split it into chunks, create embeddings,
    and store them in a FAISS vector database.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF file not found at {pdf_path}")
        return

    print(f"✅ Loading document from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Document split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_path)
    print(f"✅ Indexing complete. FAISS vector database saved at: {index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to index a PDF into a FAISS vector database.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file to index.")
    parser.add_argument("--index-path", type=str, default="faiss_index", help="Path to store the FAISS database.")

    args = parser.parse_args()
    create_faiss_index(args.pdf, args.index_path)