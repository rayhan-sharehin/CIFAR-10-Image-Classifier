# Document-Based Question Answering (Offline RAG)

## Overview
This project demonstrates a simplified Retrieval-Augmented Generation (RAG) system using offline TF-IDF embeddings and cosine similarity. It retrieves the most relevant document chunks for a question and generates an answer by combining context.

## Features
- TF-IDF vectorization of documents
- Cosine similarity retrieval (offline, no FAISS required)
- Simulated answer generation to mimic RAG pipeline
- Beginner-friendly, Kaggle-compatible

## Usage
1. Run all notebook cells
2. Update `documents` with your own data
3. Call `answer_question("Your question")` to get answers

## Technologies
- Python
- scikit-learn (TF-IDF, cosine similarity)
- NumPy

## Notes
This project simulates RAG logic in a fully offline setup, demonstrating understanding of:
- Embeddings
- Vector-based retrieval
- Contextual answer generation
