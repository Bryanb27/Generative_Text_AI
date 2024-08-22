# Generative Text AI

This repository contains the source code for a web application developed in Python that uses **Flask**, **BERT**, and **GPT-3.5-turbo** to extract text from PDF files, generate embeddings, and answer questions based on the document content.

## Features

- **Text Extraction**: Extracts text from PDF files.
- **Embedding Generation**: Uses the BERT model to generate embeddings from the extracted text.
- **Question Answering**: Integrates with the OpenAI API to answer questions based on the processed text.

## Technologies Used

- **Python**: Main programming language.
- **Flask**: Web framework used to build the application.
- **BERT**: Used for generating embeddings from the text.
- **GPT-3.5-turbo**: OpenAI model used for answering questions.
- **PyPDF2**: Library for handling PDF files.
- **Torch**: Used for working with the BERT model.

## Project Structure

```plaintext
Generative_Text_AI/
│
├── templates/               # HTML files for rendering pages
│   ├── index.html
│   └── resultado.html
│
├── PDFs/                    # Directory to store PDF files
│   └── sample.pdf
│
├── app.py                   # Main Flask application code
└── README.md                # Project documentation