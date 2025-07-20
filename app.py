# SQLite3 fix for ChromaDB on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import warnings
warnings.filterwarnings("ignore",
                        message="builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute",
                        category=DeprecationWarning)

# ==== Imports ====
import os
import tempfile
import tiktoken
from PIL import Image
from markitdown import MarkItDown
from openai import OpenAI
from llama_parse import LlamaParse
import streamlit as st

# RAG imports - Using FAISS instead of Chroma for better compatibility
import shutil
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI as LangChainOpenAI
from langchain_community.vectorstores import FAISS  # Changed from Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ==== Config ====
LLAMA_API = os.getenv("LLAMA_API_PARSE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"  # Fixed model name

# Validate API keys
if not LLAMA_API or not OPENAI_API_KEY:
    st.error("⚠️ Please set LLAMA_API_PARSE and OPENAI_API_KEY environment variables")
    st.stop()

# Title Line
st.header("💡Smart Content Extraction")
st.caption("🎲 Extract & derive any type of content with smart techniques.")

# Sidebar
logo_link = "formal image.jpg"
with st.sidebar:
    if os.path.exists(logo_link):
        try:
            logo_image = Image.open(logo_link)
            st.image(logo_image, width=150)
        except:
            st.info("Logo not found. Using default view.")
    else:
        st.info("Logo not found. Using default view.")

    st.write("## 🔗 👨‍💻 Developed By:")
    st.write("    **Eng.Ahmed Zeyad Tareq**")
    st.write("🎓 Master's in Artificial Intelligence Engineering.")
    st.write("📌 Data Scientist, AI Developer.")
    st.write("[GitHub](https://github.com/AhmedZeyadTareq) | [LinkedIn](https://www.linkedin.com/in/ahmed-zeyad-tareq) | [Kaggle](https://www.kaggle.com/ahmedzeyadtareq)")
    
    # Add clear vector store button
    if st.button("🗑️ Clear Vector Database"):
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]
            if "qa_chain" in st.session_state:
                del st.session_state["qa_chain"]
            st.success("Vector database cleared!")
            st.rerun()

# File size limit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

uploaded_file = st.file_uploader("📂 Choose File:", type=None, accept_multiple_files=True)

# ==== Functions ====

def convert_file(path: str) -> str:
    """Convert file to text (prefer structured, fallback to OCR)"""
    ext = os.path.splitext(path)[1].lower()
    try:
        print("[🔍] Trying structured text extraction via MarkItDown...")
        md = MarkItDown(enable_plugins=False)
        result = md.convert(path)
        if result.text_content.strip():
            print(f"[✔] Markdown extracted.")
            return result.text_content
        else:
            print("[⚠️] No structured text found. Fallback to OCR...")
    except Exception as e:
        print(f"[❌] MarkItDown failed: {e}. Fallback to OCR...")

    print("[🔍] OCR Started...")
    try:
        parser = LlamaParse(api_key=LLAMA_API, result_type="markdown")
        documents = parser.load_data(path)

        if not documents:
            st.error("Failed to parse the document - no content returned")
            return ""

        return documents[0].text

    except Exception as e:
        st.error(f"Error parsing document: {str(e)}")
        return ""

def reorganize_markdown(raw: str) -> str:
    """Reorganize markdown via OpenAI"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": f"reorganize the following content:\n {raw}"},
                {"role": "system", "content": (
                    "You are a reorganizer. Return the content in Markdown, keeping it identical. "
                    "Do not delete or replace anything—only reorganize for better structure. your response the content direct without (``` ```)."
                )}
            ]
        )
        print("===Reorganized Done===")
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error reorganizing content: {str(e)}")
        return raw

def add_to_vectorstore(content: str):
    """Add content to existing vector store or create new one using FAISS"""
    try:
        # Create document from content
        doc = Document(page_content=content)
        
        # Split the document into chunks
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents([doc])
        print(f"Adding {len(chunks)} chunks to vector store")
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Check if vector store already exists
        if "vectorstore" not in st.session_state:
            # Create new FAISS vector store
            st.session_state["vectorstore"] = FAISS.from_documents(
                documents=chunks, 
                embedding=embeddings
            )
            print("Created new vector store")
        else:
            # Add to existing vector store
            st.session_state["vectorstore"].add_documents(chunks)
            print("Added to existing vector store")
        
        # Setup/update QA chain with better retriever settings
        llm = LangChainOpenAI(temperature=0, model=LLM_MODEL, api_key=OPENAI_API_KEY)
        retriever = st.session_state["vectorstore"].as_retriever(
            search_kwargs={"k": 5}  # Retrieve 5 chunks for better context
        )
        st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever, 
            chain_type="stuff", 
            verbose=True
        )
    except Exception as e:
        st.error(f"Error adding to vector store: {str(e)}")

def rag(question: str) -> str:
    """Answer questions from vector database"""
    if "qa_chain" not in st.session_state:
        st.error("No content in vector database. Please extract files first.")
        return "No content available for querying."
    
    try:
        response = st.session_state["qa_chain"].invoke({"query": question})
        return response['result']
    except Exception as e:
        st.error(f"Error querying: {str(e)}")
        return "Error processing your question. Please try again."

def count_tokens(content: str, model="gpt-4o-mini"):
    """Count tokens in the content"""
    try:
        enc = tiktoken.encoding_for_model(model)
        token_count = len(enc.encode(content))
        print(f"The Size of the Content_Tokens: {token_count}")
        return token_count
    except:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(content))
        print(f"The Size of the Content_Tokens: {token_count}")
        return token_count

# ==== Main Process ====

if uploaded_file:
    for i, file in enumerate(uploaded_file):
        # Check file size
        if file.size > MAX_FILE_SIZE:
            st.error(f"❌ {file.name} exceeds 10MB limit. Please upload a smaller file.")
            continue
            
        st.write(f"**File {i+1}: {file.name}** ({file.size / 1024:.1f} KB)")
        suffix = os.path.splitext(file.name)[1]
        
        if st.button(f"Start 🔁 - {file.name}", key=f"start_{i}"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file.getvalue())
                file_path = tmp_file.name
                
            try:
                with st.spinner(f"Processing {file.name}..."):
                    raw_text = convert_file(file_path)
                    
                    if raw_text:
                        # Count tokens
                        token_count = count_tokens(raw_text)
                        st.info(f"📊 Token count: {token_count}")
                        
                        st.text_area(f"📄 Content from {file.name}:", raw_text, height=200, key=f"content_{i}")
                        
                        # Automatically add to vector store after extraction
                        add_to_vectorstore(raw_text)
                        st.success(f"✅ Content from {file.name} added to vector database!")
                        
                        # Store in session state for reorganization
                        if f"raw_text_{i}" not in st.session_state:
                            st.session_state[f"raw_text_{i}"] = []
                        st.session_state[f"raw_text_{i}"].append(raw_text)
                    else:
                        st.error(f"Failed to extract content from {file.name}")
            finally:
                # Clean up temp file
                if os.path.exists(file_path):
                    os.unlink(file_path)

        if f"raw_text_{i}" in st.session_state and st.session_state[f"raw_text_{i}"]:
            if st.button(f"🧹 Reorganize Content - {file.name}", key=f"reorganize_{i}"):
                with st.spinner("Reorganizing content..."):
                    organized = reorganize_markdown(st.session_state[f"raw_text_{i}"][-1])
                    st.markdown(organized)
                    st.download_button(
                        label=f"⬇️ Download {file.name} as TXT",
                        data=organized,
                        file_name=f"reorganized_{file.name}.txt",
                        mime="text/plain",
                        key=f"download_txt_{i}"
                    )

# Question section (available if any content is in vector store)
if "vectorstore" in st.session_state:
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About All Uploaded Content")
    
    question = st.text_input("Ask Anything about Content..❓")
    if st.button("💬 Send"):
        if question:
            with st.spinner("Searching for answer..."):
                answer = rag(question)
                st.markdown(f"**Question❓:**\n{question}")
                st.markdown(f"**Answer💡:**\n{answer}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("👆 Upload and process files to enable Q&A functionality")
