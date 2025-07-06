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

# ==== Config ====
LLAMA_API = os.getenv("LLAMA_API_PARSE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4.1-mini"

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

uploaded_file = st.file_uploader("📂 Choose File:", type=None)

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


def rag(con: str, question: str) -> str:
    """Answer questions from provided content"""
    client = OpenAI()
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": question},
            {"role": "system", "content": f"You are an assistant. Answer concisely from the following content:\n {con}"}
        ]
    )
    return completion.choices[0].message.content


def count_tokens(content: str, model="gpt-4-turbo"):
    """Count tokens in the content"""
    enc = tiktoken.encoding_for_model(model)
    print(f"The Size of the Content_Tokens: {len(enc.encode(content))}")


# ==== Main Process ====

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

        if st.button("Start 🔁"):
            raw_text = convert_file(file_path)
            st.text_area("📄 Content:", raw_text, height=200)
            st.session_state["raw_text"] = raw_text

        if "raw_text" in st.session_state:
            if st.button("🧹 Reorganize Content"):
                organized = reorganize_markdown(st.session_state["raw_text"])
                st.session_state["organized_text"] = organized
                st.markdown(organized)
                st.download_button(
                    label="⬇️ Download as TXT",
                    data=organized,
                    file_name="reorganized_content.txt",
                    mime="text/plain",
                    key="download_txt"
                )
                
            #if "organized_text" in st.session_state:
            question = st.text_input("Ask Anything about Content..❓")
            if st.button("💬 Send"):
                content_to_use = st.session_state.get("organized_text", st.session_state["raw_text"])
                answer = rag(content_to_use, question)
                st.markdown(f"**Question❓:**\n{question}")
                st.markdown(f"**Answer💡:**\n{answer}")
                    
