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

# RAG imports
import shutil
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI as LangChainOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

# ==== Config ====
LLAMA_API = os.getenv("LLAMA_API_PARSE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"  # Fixed model name

# Title Line
st.header("💡Smart Content Extraction")
# Description Line
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
    
    # Clear vector store button
    if st.button("🗑️ Clear Vector Database"):
        if "vectorstore_index" in st.session_state:
            # Clean up temporary files
            if "loaded_files" in st.session_state:
                for file_path in st.session_state["loaded_files"]:
                    if os.path.exists(file_path) and file_path.startswith("temp_"):
                        try:
                            os.unlink(file_path)
                        except:
                            pass
            
            # Clear session state
            for key in ["vectorstore_index", "llm", "loaded_files"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.success("Vector database cleared!")
            st.rerun()

uploaded_file = st.file_uploader("📂 Choose File:", type=None, accept_multiple_files=True)

#####################################
#####################################

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
        st.error(f"Error reorganizing: {str(e)}")
        return raw

def add_to_vectorstore(content: str, filename: str = "temp_content.txt"):
    """Add content to vector store using VectorstoreIndexCreator"""
    try:
        # Save content to a temporary file (TextLoader needs a file path)
        temp_file_path = f"temp_{filename}"
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Load the document using LangChain's TextLoader
        text_loader = TextLoader(temp_file_path, encoding="utf-8")
        
        # Initialize OpenAI embedding model
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Create text splitter with better settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create a vector index from the document with explicit vectorstore
        if "vectorstore_index" not in st.session_state:
            # Create new index with FAISS as the vectorstore
            st.session_state["vectorstore_index"] = VectorstoreIndexCreator(
                embedding=embeddings,
                text_splitter=text_splitter,
                vectorstore_cls=FAISS  # Explicitly specify FAISS to avoid the warning
            ).from_loaders([text_loader])
            st.session_state["loaded_files"] = [temp_file_path]
            print("Created new vector store index")
        else:
            # For multiple files, we need to recreate the index with all files
            st.session_state["loaded_files"].append(temp_file_path)
            
            # Create loaders for all files
            all_loaders = []
            for file_path in st.session_state["loaded_files"]:
                if os.path.exists(file_path):
                    all_loaders.append(TextLoader(file_path, encoding="utf-8"))
            
            # Recreate index with all documents
            st.session_state["vectorstore_index"] = VectorstoreIndexCreator(
                embedding=embeddings,
                text_splitter=text_splitter,
                vectorstore_cls=FAISS
            ).from_loaders(all_loaders)
            print(f"Updated vector store index with {len(all_loaders)} documents")
        
        # Store the LLM for later use - Use LangChainOpenAI, not OpenAI
        st.session_state["llm"] = LangChainOpenAI(
            temperature=0.1, 
            model=LLM_MODEL, 
            api_key=OPENAI_API_KEY
        )
        
    except Exception as e:
        st.error(f"Error adding to vector store: {str(e)}")
        # Clean up on error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def rag(question: str) -> str:
    """Answer questions from vector database using query_with_sources"""
    if "vectorstore_index" not in st.session_state:
        st.error("No content in vector database. Please extract files first.")
        return "No content available for querying."
    
    try:
        # Get the index and LLM
        index = st.session_state["vectorstore_index"]
        llm = st.session_state["llm"]
        
        # Add prompt prefix for better answers
        prompt_query = "Answer directly and concise and specific:\n" + question
        
        # Perform retrieval-based query
        response = index.query_with_sources(prompt_query, llm=llm)
        
        # The response contains 'answer' and 'sources'
        answer = response.get('answer', 'No answer found.')
        sources = response.get('sources', '')
        
        # Format response with sources if available
        if sources and sources.strip():
            return f"{answer}\n\n📚 Sources: {sources}"
        else:
            return answer
            
    except Exception as e:
        st.error(f"Error querying: {str(e)}")
        return "Error processing your question."

def count_tokens(content: str, model="gpt-4o-mini"):
    """Count tokens in the content"""
    try:
        enc = tiktoken.encoding_for_model(model)
        token_count = len(enc.encode(content))
        print(f"The Size of the Content_Tokens: {token_count}")
        return token_count
    except:
        # Fallback for unknown models
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(content))
        print(f"The Size of the Content_Tokens: {token_count}")
        return token_count

# ==== Main Process ====

if uploaded_file:
    for i, file in enumerate(uploaded_file):
        st.write(f"**File {i+1}: {file.name}**")
        suffix = os.path.splitext(file.name)[1]
        
        if st.button(f"Start 🔁 - {file.name}", key=f"start_{i}"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file.getvalue())
                file_path = tmp_file.name
            
            try:
                with st.spinner(f"Processing {file.name}..."):
                    raw_text = convert_file(file_path)
                    
                    if raw_text:
                        # Count and display tokens
                        token_count = count_tokens(raw_text)
                        st.info(f"📊 Token count: {token_count}")
                        
                        st.text_area(f"📄 Content from {file.name}:", raw_text, height=200, key=f"content_{i}")
                        
                        # Automatically add to vector store after extraction
                        raw = reorganize_markdown(raw_text)
                        add_to_vectorstore(raw_text, filename=file.name)
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
if "vectorstore_index" in st.session_state:  # Fixed: was checking for "vectorstore"
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
