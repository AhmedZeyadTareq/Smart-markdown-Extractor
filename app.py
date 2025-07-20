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
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI as LangChainOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ==== Config ====
LLAMA_API = os.getenv("LLAMA_API_PARSE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4.1-mini"

# Clean up ChromaDB on app start
shutil.rmtree("chroma_DB", ignore_errors=True)

# Title Line (Slightly smaller than st.title)
st.header("💡Smart Content Extraction")
# Description Line (Subtle text)
st.caption("🎲 Extract & derive any type of content with smart techniques.")

# Sidebar
logo_link = "formal image.jpg"
with st.sidebar:
    if os.path.exists(logo_link):
        logo_image = Image.open(logo_link)
        st.image(logo_image, width=150)
    else:
        st.warning("Logo not found. Please check the logo path.")

    st.write("## 🔗 👨‍💻 Developed By:")
    st.write("    **Eng.Ahmed Zeyad Tareq**")
    st.write("🎓 Master's in Artificial Intelligence Engineering.")
    st.write("📌 Data Scientist, AI Developer.")
    st.write("[GitHub](https://github.com/AhmedZeyadTareq) | [LinkedIn](https://www.linkedin.com/in/ahmed-zeyad-tareq) | [Kaggle](https://www.kaggle.com/ahmedzeyadtareq)")

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
            # with open('data.md', 'a', encoding='utf-8') as f:
            #     f.write(result.text_content)
            return result.text_content
        else:
            print("[⚠️] No structured text found. Fallback to OCR...")
    except Exception:
        print(f"[❌] MarkItDown failed. Fallback to OCR...")

    print("[🔍] OCR Started...")
    try:
        # Initialize the parser
        parser = LlamaParse(api_key=LLAMA_API, result_type="markdown")

        # Parse the file using its path
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
    client = OpenAI()
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
    # with open('data.md', 'a', encoding='utf-8') as f:
    #     f.write(completion.choices[0].message.content)
    print("===Reorganized Done===")
    return completion.choices[0].message.content

def add_to_vectorstore(content: str):
    """Add content to existing vector store or create new one"""
    # Create document from content
    doc = Document(page_content=content)
    
    # Split the document into chunks
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    print(f"Adding {len(chunks)} chunks to vector store")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Check if vector store already exists
    if "vectorstore" not in st.session_state:
        # Create new vector store
        st.session_state["vectorstore"] = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory="chroma_DB"
        )
        print("Created new vector store")
    else:
        # Add to existing vector store
        st.session_state["vectorstore"].add_documents(chunks)
        print("Added to existing vector store")
    
    # Setup/update QA chain
    llm = LangChainOpenAI(temperature=0, model="gpt-4o-mini")
    retriever = st.session_state["vectorstore"].as_retriever()
    st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        chain_type="stuff", 
        verbose=True
    )

def rag(question: str) -> str:
    """Answer questions from vector database"""
    if "qa_chain" not in st.session_state:
        st.error("No content in vector database. Please extract files first.")
        return "No content available for querying."
    
    response = st.session_state["qa_chain"].invoke({"query": question})
    return response['result']


def count_tokens(content: str, model="gpt-4-turbo"):
    """Count tokens in the content"""
    enc = tiktoken.encoding_for_model(model)
    print(f"The Size of the Content_Tokens: {len(enc.encode(content))}")


# ==== Main Process ====

if uploaded_file:
    for i, file in enumerate(uploaded_file):
        st.write(f"**File {i+1}: {file.name}**")
        suffix = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.getvalue())
            file_path = tmp_file.name

            if st.button(f"Start 🔁 - {file.name}", key=f"start_{i}"):
                raw_text = convert_file(file_path)
                st.text_area(f"📄 Content from {file.name}:", raw_text, height=200, key=f"content_{i}")
                
                # Automatically add to vector store after extraction
                add_to_vectorstore(raw_text)
                st.success(f"✅ Content from {file.name} added to vector database!")
                
                # Store in session state for reorganization
                if f"raw_text_{i}" not in st.session_state:
                    st.session_state[f"raw_text_{i}"] = []
                st.session_state[f"raw_text_{i}"].append(raw_text)

            if f"raw_text_{i}" in st.session_state and st.session_state[f"raw_text_{i}"]:
                if st.button(f"🧹 Reorganize Content - {file.name}", key=f"reorganize_{i}"):
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
            answer = rag(question)
            st.markdown(f"**Question❓:**\n{question}")
            st.markdown(f"**Answer💡:**\n{answer}")
        else:
            st.warning("Please enter a question.")
