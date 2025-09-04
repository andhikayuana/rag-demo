import os
import gradio as gr
import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama

# --- Initialize the LLM client ---
# Option 1: LM Studio
# Use http://127.0.0.1:1234 if running locally on the same machine
llm = ChatOpenAI(openai_api_base="http://127.0.0.1:1234/v1", openai_api_key="not-needed")

# Option 2: Ollama
# Replace "mistral" with the model name you downloaded in Ollama
# llm = Ollama(base_url="http://127.0.0.1:11434", model="mistral")

# --- Load the vector database and build the RAG chain ---
def setup_rag_chain(index_path: str):
    """Load the FAISS vector database and create a RAG chain."""
    if not os.path.exists(index_path):
        print(f"❌ ERROR: Vector database not found at {index_path}. Run ingest.py first.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ FAISS vector database loaded from {index_path}.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2})
    )
    return qa_chain

# --- Chat function ---
def chat_with_pdf(query, rag_chain):
    """Function called by Gradio to generate answers."""
    if rag_chain is None:
        return "The RAG system is not ready. Please check that the vector database has been created."

    response = rag_chain.invoke({'query': query})
    return response['result']

# --- Run the web application ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web chat app with your PDF using local RAG.")
    parser.add_argument("--index-path", type=str, default="faiss_index", help="Path to the FAISS database.")
    args = parser.parse_args()

    # Initialize the RAG chain once when the app starts
    qa_chain = setup_rag_chain(args.index_path)

    # Build the UI with Gradio
    iface = gr.Interface(
        fn=lambda q: chat_with_pdf(q, qa_chain),
        inputs=gr.Textbox(lines=2, placeholder="Ask something about your PDF..."),
        outputs="text",
        title="Chat with Your PDF (Local)",
        description="This RAG system answers questions based on the indexed PDF content.",
    )

    iface.launch()