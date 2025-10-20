# Ollama-RAG-Chatbox

## Overview  
This is a Retrieval-Augmented Generation (RAG) chatbot built using Streamlit to answer user questions based on uploaded documents. Users can upload CSV, Excel, or PDF files, and the bot will extract relevant information from those documents to answer user questions. If the context from the documents is insufficient, the chatbot will inform the user.

## ⚙️ Tech Stack

Frontend: Streamlit

Data Handling: Pandas, PyPDF

Embeddings: HuggingFace all-MiniLM-L6-v2

Vector Database: Chroma

LLM: Ollama qwen2

Language: Python

## 🌟 Key Features

Automated Data Parsing: Detects and processes CSV, Excel, or PDF formats into structured text.

Text Chunking Engine: Splits long text dynamically for efficient retrieval without losing context.

Vector Embedding & Search: Converts text into embeddings and retrieves top-ranked chunks using semantic similarity.

Context-Grounded QA: Generates responses strictly based on uploaded content using Qwen2 via Ollama.

Transparent References: Displays the exact document sections used to support each answer.

