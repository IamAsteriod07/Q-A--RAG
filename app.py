"""
Q&A Mini RAG system.
Provides CLI interface for running different components.
"""
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import time
import pickle

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Set up Streamlit title
st.title("Document Q&A with RAG")

# Initialize GROQ model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Chat prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")

# Initialize vector search and embeddings from documents (only when explicitly requested)
def initialize_vector_search():
    # Avoid re-initializing during the same session
    if st.session_state.get("initialized"):
        st.write("Vector store already initialized in this session.")
        return

    # If vectors exist in session state, assume initialized
    if "vectors" in st.session_state:
        st.session_state["initialized"] = True
        st.write("Vector store available in session.")
        return

    # Check if cached vectors exist on disk
    try:
        with open("vectors.pkl", "rb") as f:
            st.session_state.vectors = pickle.load(f)
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state["initialized"] = True
            st.write("Loaded vectors from cache.")
            return
    except FileNotFoundError:
        pass

    # Initialize embeddings
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load PDFs from folder (documents only — do NOT add user prompts)
    loader = PyPDFDirectoryLoader("./industrial")
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])

    # Create FAISS vector store from documents
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

    # Cache vectors locally
    with open("vectors.pkl", "wb") as f:
        pickle.dump(st.session_state.vectors, f)

    st.session_state["initialized"] = True
    st.write("Vector Store DB is ready.")

# Streamlit input for user question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to run document embedding (initialize only once)
if st.button("Documents Embedding"):
    initialize_vector_search()

# Handle question answering
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please embed documents first!")
    else:
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Create retriever
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Run the chain and measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")

        # Display answer
        st.write(response['answer'])

        # Display document similarity search
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------------------------------")
