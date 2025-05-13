# Retrieval-Augmented Generation (RAG) System

## Project Description

This project implements a website-based RAG system that combines data from live web scraping and uploaded PDF documents. The web application allows users to input keywords to scrape WeChat articles, upload their own PDFs, and ask questions based on the collected content. The system uses FAISS for vector storage and the ChatGLM API to answer user queries.

## Features

- **Web Scraping**: Scrape WeChat articles using Selenium based on user-provided keywords
- **PDF Upload and Text Extraction**: Upload PDF files and extract text content using PyMuPDF
- **FAISS Vector Index**: Chunk text into sections and generate vector embeddings for efficient similarity search
- **Question Answering**: Retrieve relevant text chunks using FAISS and generate answers with the ChatGLM API

## Project Structure
```
RAG System/
├── app.py                     # Main Flask application
├── scraper.py                 # WeChat scraping module
├── templates/
│   └── index.html             # Main webpage
├── static/
│   └── style.css              # CSS styling
├── uploads/                   # Store uploaded PDFs
├── scraped/                   # Store temporary scraped articles
└── FAISS files/               # Store FAISS index and related files
```

## Run Project

```bash
python app.py
```
Open your browser and navigate to http://127.0.0.1:5000
