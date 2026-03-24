# Research Paper Insight Extractor

A powerful Streamlit web application that uses AI to analyze research papers and extract key insights, summaries, and answer questions about the content.

## Features

- **PDF Upload**: Drag-and-drop or browse to upload research papers
- **AI-Powered Summary**: Generate comprehensive summaries of research papers
- **Key Insight Extraction**: Extract methodology, contributions, limitations, and future research directions
- **Interactive Q&A Chat**: Ask questions about the paper using RAG (Retrieval Augmented Generation)
- **Export Options**: Download summaries, insights, and chat history as PDF or TXT
- **Dark/Light Mode**: Toggle between themes for comfortable reading
- **Paper Metadata Detection**: Automatically extract title, authors, abstract, keywords, and DOI

## Project Structure

```
project/
│
├── app.py                      # Main Streamlit application
├── utils/
│   ├── __init__.py             # Package initialization
│   ├── pdf_parser.py           # PDF text extraction utility
│   ├── summarizer.py           # LLM-based summarization module
│   ├── insight_extractor.py    # Key insight extraction module
│   └── chat_engine.py          # RAG-based Q&A chat engine
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Step 1: Clone or Download the Project

```bash
cd "AI Research Paper"
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables (Optional)

You can set your OpenAI API key as an environment variable:

```bash
# On Windows (PowerShell):
$env:OPENAI_API_KEY="your-api-key-here"

# On Windows (CMD):
set OPENAI_API_KEY=your-api-key-here

# On macOS/Linux:
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage Guide

### 1. Upload a Research Paper

1. Click on the upload area or drag-and-drop a PDF file
2. The app will automatically extract text and detect metadata
3. Review the extracted paper information (title, authors, abstract, etc.)

### 2. Analyze the Paper

Click the "Analyze Paper" button to:
- Generate a comprehensive summary
- Extract key insights
- Identify methodology, contributions, limitations, and future research

### 3. Explore Results

Navigate through the tabs:
- **Summary**: View the AI-generated summary
- **Key Insights**: Explore detailed findings, methodology, contributions, limitations
- **Q&A Chat**: Ask questions about the paper
- **Downloads**: Export your results

### 4. Ask Questions

In the Q&A Chat tab:
- Use suggested questions or type your own
- The AI will answer based only on the paper content
- Chat history is maintained during the session

### 5. Download Results

Export your analysis as:
- TXT files for plain text
- PDF files for formatted documents
- Full report combining all sections

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| PDF Processing | pdfplumber |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| PDF Export | ReportLab |

## Configuration Options

### Model Selection

The application uses `gpt-4o-mini` by default for cost-effective processing. To use a different model, modify the `model` variable in:
- `utils/summarizer.py`
- `utils/insight_extractor.py`
- `utils/chat_engine.py`

### Chunk Size Settings

For large papers, you can adjust chunking parameters in `utils/chat_engine.py`:
- `chunk_size`: Characters per chunk (default: 500)
- `chunk_overlap`: Overlap between chunks (default: 100)
