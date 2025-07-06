# 💡 Smart Content Extraction

A powerful Streamlit application that intelligently extracts and processes content from various file formats using advanced AI techniques. This tool combines structured text extraction with OCR capabilities and provides intelligent content reorganization and question-answering features.

## 🎯 Overview

Smart Content Extraction is designed to handle diverse file formats and extract meaningful content using a two-tier approach:
1. **Structured Extraction**: Uses MarkItDown for direct text extraction from supported formats
2. **OCR Fallback**: Employs LlamaParse for optical character recognition when structured extraction fails
3. **AI Enhancement**: Leverages OpenAI's GPT models for content reorganization and intelligent Q&A

## ✨ Features

- 📂 **Universal File Support**: Works with multiple file formats including PDFs, images, documents, and more
- 🔍 **Smart Extraction**: Intelligent fallback from structured to OCR-based extraction
- 🧹 **Content Reorganization**: AI-powered content restructuring for better readability
- 💬 **Interactive Q&A**: Ask questions about your extracted content using RAG (Retrieval-Augmented Generation)
- ⬇️ **Export Options**: Download reorganized content as text files
- 📊 **Token Counting**: Monitor content size for API usage optimization

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- API keys for:
  - OpenAI API
  - LlamaParse API

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AhmedZeyadTareq/Smart-markdown-Extractor.git
   cd Smart-markdown-Extractor
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up .env file with API keys:**
   ```bash
   OPENAI_API_KEY="your-openai-api-key"
   LLAMA_API_PARSE="your-llamaparse-api-key"
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📋 Dependencies

```
streamlit
openai
llama-parse
markitdown
pillow
tiktoken
```

## 🚀 Usage

### Method 1: Local Development
1. Follow the installation steps above
2. Run `streamlit run app.py`
3. Open your browser to `http://localhost:8501`

### Method 2: Deployed Version
Access the live application at: [Try it live](https://contenttomarkdownocr-ahmedtareq.streamlit.app/)

### Method 3: import the function in another code (Optional)
```bash
from app import convert_file, reorganize_markdown, rag

md_content = convert_file("document.pdf")
organized_md = reorganize_markdown(md_content)
answer = rag(organized_md, "What is this document about?")
print(answer)
```

## 📖 How to Use

1. **Upload File**: Click "📂 Choose File" and select your document
2. **Extract Content**: Click "Start 🔁" to begin extraction
3. **Reorganize** (Optional): Click "🧹 Reorganize Content" for AI-enhanced formatting
4. **Ask Questions**: Use the text input to ask questions about your content
5. **Download**: Save the reorganized content using the download button

## 🔧 Configuration

### API Configuration
- **OpenAI Model**: Currently set to `gpt-4.1-mini` (configurable in `LLM_MODEL`)
- **LlamaParse**: Uses markdown output format for better structure

### Customization Options
- Modify `LLM_MODEL` variable to use different OpenAI models
- Adjust the reorganization prompt in the `reorganize_markdown()` function
- Customize the RAG system prompt in the `rag()` function

## 📁 Project Structure

```
smart-content-extraction/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── formal image.jpg      # Logo image (optional)
└── .env                  # Environment variables (not tracked)
```

## 🎯 Use Cases

- **Document Analysis**: Extract and analyze content from research papers, reports, and presentations
- **Data Processing**: Convert scanned documents and images to searchable text
- **Content Creation**: Reorganize and structure extracted content for better readability
- **Research Assistant**: Ask questions about document content using natural language
- **Batch Processing**: Handle multiple documents with consistent extraction quality

## 🔍 Technical Details

### Extraction Pipeline
1. **Primary Method**: MarkItDown attempts structured extraction
2. **Fallback Method**: LlamaParse handles OCR when structured extraction fails
3. **Content Processing**: OpenAI GPT models enhance and reorganize content
4. **Interactive Layer**: RAG system enables intelligent question-answering

### Error Handling
- Graceful fallback between extraction methods
- Comprehensive error messages for debugging
- Robust file handling with temporary file management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Known Issues

- Large files may take longer to process due to API rate limits
- Some complex layouts might require manual review after extraction
- OCR accuracy depends on image quality and text clarity

## 📝 Changelog

### v1.0.0
- Initial release with basic extraction and reorganization features
- Integrated MarkItDown and LlamaParse for robust content extraction
- Added interactive Q&A functionality using RAG

---

## 👨‍💻 Developed By
### **Ahmed Zeyad Tareq**  
📌 Data Scientist & AI Developer | 🎓 Master of AI Engineering
- 📞 WhatsApp: +905533333587 
- [GitHub](https://github.com/AhmedZeyadTareq) | [LinkedIn](https://www.linkedin.com/in/ahmed-zeyad-tareq) | [Kaggle](https://www.kaggle.com/ahmedzeyadtareq)

## 📄 License
MIT License © Ahmed Zeyad Tareq

---

⭐ If you find this project useful, please give it a star on GitHub!
