# 🤖 AI Document-Based Question Generator (Web)

An end-to-end **RAG + LLM–powered question generation** system with a **web interface**, combining:
- Automated **document ingestion** (PDF/DOCX → text).
- **Semantic chunking** + TF-IDF keyword extraction.
- **Embeddings** with **SentenceTransformer (MiniLM)**.
- **Vector search retrieval** powered by **FAISS**.
- **Few-shot LLM prompting** to generate **MCQ**, **Yes/No**, **Descriptive**, and **Coding** questions.
- **Post-processing pipeline** including parsing, structural validation, **semantic deduplication**, and difficulty grouping.
- Optional **answer evaluation** for non-coding questions.

The frontend (HTML/JS) is served via **Node.js**, and the backend is implemented in **Python**, integrating the complete question-generation pipeline.

---

## 📚 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Quick Start (Web UI)](#-quick-start-web-ui)
- [Repository Structure](#-repository-structure)
- [Requirements](#-requirements)
  - [1) Python](#1-python)
  - [2) Python Packages](#2-python-packages)
  - [3) Node.js Setup](#3-nodejs-setup)
  - [4) Environment Variables](#4-environment-variables)
- [Download the Code](#-download-the-code)
- [Run (Web UI)](#-run-web-ui)
- [Core Pipeline Explanation](#-core-pipeline-explanation)
  - [Document Extraction](#document-extraction)
  - [Chunking](#chunking)
  - [TF-IDF Keyword Extraction](#tf-idf-keyword-extraction)
  - [Embeddings & FAISS Retrieval](#embeddings--faiss-retrieval)
  - [LLM Question Generation](#llm-question-generation)
  - [Post-Processing (Parsing, Validation, Deduplication)](#post-processing-parsing-validation-deduplication)
  - [Difficulty Control](#difficulty-control)
  - [Evaluation](#evaluation)
- [Config You May Need to Edit](#-config-you-may-need-to-edit)
- [Troubleshooting](#-troubleshooting)
- [Security Note](#-security-note)
- [Sample requirements.txt](#-sample-requirementstxt)
- [Demonstration](#-demonstration)

---

## 🔎 Overview
This repository provides a **full-stack web application** where users upload documents, choose question types and difficulty levels, and receive fully AI-generated questions grounded in the original content. It wraps the entire backend RAG pipeline behind a clean Node-served UI.

---

## ✨ Features
- **Upload PDFs or DOCX** files → automatically extracted via `PyMuPDF` or `python-docx`.
- **Custom character-based chunking** (no LangChain).
- **TF-IDF keyword extraction** to highlight key topics.
- **SentenceTransformer embeddings** (`all-MiniLM-L6-v2`).
- **FAISS vector database** for fast semantic retrieval.
- **Few-shot LLM prompting** for:
  - MCQ  
  - Yes/No  
  - Descriptive  
  - Coding  
- **Difficulty-aware generation**: Easy → Medium → Hard.
- **Post-generation cleaning**:
  - Parsing  
  - Structural rule validation  
  - **Semantic deduplication**  
  - Final assembly into a test set  
- Optional **evaluation** for user answers.

---

## ▶️ Quick Start (Web UI)

1. **Clone**
    ```
    git clone https://github.com/<your-account>/<your-repo>.git
    cd <your-repo>/prototype
    ```

2. **Create & activate Python environment**
    ```
    python -m venv .venv
    source .venv/bin/activate        # macOS/Linux
    # .\.venv\Scripts\Activate.ps1    # Windows
    pip install -r requirements.txt
    ```

3. **Install Node packages**
    ```
    npm install
    ```

4. **Set environment variables**
    ```
    export OPENAI_API_KEY="sk-..."
    export EMBED_MODEL_NAME="all-MiniLM-L6-v2"
    export BACKEND_PORT=5000
    export FRONTEND_PORT=3000
    ```

5. **Run backend**
    ```
    python generator.py
    ```

6. **Run frontend**
    ```
    npm start
    ```

7. **Open** `http://localhost:3000` to use the tool.

---

## 📁 Repository Structure
```
prototype/
├─ public/
│ ├─ index.html # main UI
│ ├─ middle.png # assets
│ └─ start.png
├─ generator.py # backend API — full RAG pipeline
├─ report.py # evaluation + report formatting
├─ server.js # Node.js server
├─ package.json
├─ requirements.txt
├─ Check_3.ipynb # development notebook
├─ assignread.txt # notes
└─ README.md
```

---

## ✅ Requirements

### 1) Python
- Python **3.8+**  
- Works on Windows / macOS / Linux

### 2) Python Packages  
Install via:
```
pip install -r requirements.txt
```
```

Typical dependencies:
- numpy  
- scikit-learn  
- sentence-transformers  
- faiss-cpu  
- pymupdf  
- python-docx / docx2txt  
- flask / flask-cors  
- openai  
- tqdm  
```
### 3) Node.js Setup
`
npm install
npm start
`


### 4) Environment Variables
`
OPENAI_API_KEY=sk-...
EMBED_MODEL_NAME=all-MiniLM-L6-v2
BACKEND_PORT=5000
FRONTEND_PORT=3000
`

---

## ⬇️ Download the Code

`
git clone https://github.com/<your-account>/<your-repo>.git
cd <your-repo>/prototype
`

---

## ▶️ Run (Web UI)

Backend
```
python generator.py
```
Frontend
```
npm start
```


---

# 🧠 Core Pipeline Explanation

## **Document Extraction**
- For **PDF**, uses `PyMuPDF (fitz)` → extracts plain text.  
- For **DOCX**, uses `python-docx` / `docx2txt`.  
- All extracted text is merged into **one continuous normalized string**.

---

## **Chunking**
- Custom **character-based sliding window**.
- Example:
  - `CHUNK_SIZE = 3200`
  - `OVERLAP = 300`
- Produces overlapping chunks to preserve meaning.
- No external splitting tools used.

---

## **TF-IDF Keyword Extraction**
- `TfidfVectorizer` analyzes all chunks.
- TF = term frequency in a chunk.  
- IDF = rarity across all chunks.  
- Extracts **top 3–6 distinctive keywords** per chunk.
- Keywords guide retrieval and improve generation grounding.

---

## **Embeddings & FAISS Retrieval**
- Embedding model: **`all-MiniLM-L6-v2`** (SentenceTransformer).  
- Embeds every chunk → vector database of embeddings.  
- FAISS used for **nearest-neighbor similarity**.  
- Retrieval query is built from **keywords**, *not* from question type.  
- Retrieves **top-k = 4** relevant chunks per query.

---

## **LLM Question Generation**
For each question type:
1. Retrieve top-k chunks.  
2. Build prompt with:
   - Context chunks  
   - TF-IDF keywords  
   - Difficulty rubric  
   - Type instruction (MCQ, Yes/No, etc.)  
   - Few-shot examples  
3. LLM generates multiple question candidates (over-generation).  
4. Candidates stored for post-processing.

---

## **Post-Processing (Parsing, Validation, Deduplication)**

### **Parsing**
- Convert raw LLM output → `{question, options, answer, explanation}`.

### **Validation**
- MCQ → must have 4 non-empty options + 1 correct answer.  
- Yes/No → answer must be strictly “Yes” or “No”.  
- Descriptive → must include model answer.  
- Coding → must include code-oriented problem description.

### **Semantic Deduplication**
- Embed each question using MiniLM.  
- Sort by quality score.  
- Greedy selection:
  - Keep highest-quality question first.  
  - For each new question:  
    - Compute cosine similarity  
    - If similarity > **0.8**, discard  
- Ensures a unique, non-redundant final set.

---

## **Difficulty Control**
- User specifies: number or percentage for *Easy / Medium / Hard*.  
- Converted into concrete counts.  
- For each difficulty bucket:
  - Uses a **difficulty-specific prompt template**.  
  - Generates multiple candidates.  
  - Tags each candidate with difficulty label.  
  - Post-processing selects validated questions to satisfy the quota.  
- If not enough valid questions → returns fewer (no auto-regen loop).

---

## **Evaluation**
- MCQ / Yes-No → direct correctness check.  
- Descriptive → LLM-based scoring rubric.  
- Coding → flagged for manual/sandbox evaluation.  
- `report.py` outputs structured feedback.

---

## ⚙️ Config You May Need to Edit
- `CHUNK_SIZE`, `CHUNK_OVERLAP`  
- `TOP_K`  
- `DEDUP_SIM_THRESHOLD`  
- Embedding model name  
- LLM temperature / n-completions  
- Backend & frontend port numbers  
- API keys via `.env`  

---

## 🧩 Troubleshooting

- **FAISS install error (Windows)**  
  → use `faiss-cpu`.  
- **Invalid LLM formats**  
  → tighten few-shot examples.  
- **Duplicate questions**  
  → lower similarity threshold (0.75).  
- **Extraction issues**  
  → ensure PDFs are text-based (not scanned).  
- **CORS issues**  
  → configure Flask/Node CORS headers.

---

## 🔐 Security Note
- Never commit API keys.  
- Use `.env` + `.gitignore`.  
- Uploaded documents may contain sensitive text — treat carefully.  
- Sandbox code evaluation if you extend coding features.

---

## 📄 Sample requirements.txt
```
numpy
faiss-cpu
sentence-transformers
pymupdf
python-docx
docx2txt
scikit-learn
tqdm
flask
flask-cors
openai
```

---

## 🎥 Demonstration
**Demo video:** `https://drive.google.com/file/d/1rLJVi1iD4d8YMxG5_na1w9aCELY3unUX/view?usp=sharing`



---
