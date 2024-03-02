#chatWithSciBot

## Introduction
------------
SciBot is a Python application that allows you to chat with multiple documents in multiple formats. You can ask questions about the documents using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. It supports PDFs, PowerPoint presentations, DOCX files, TXT files, LaTeX files, CSV files, Excel files, and PNG images.Please note that the app will only respond to questions related to the loaded documents.

## How It Works
------------

![SciBot Architecture Diagram](./architecturelangchain.png)

The application follows these steps to provide responses to your questions:

1. *Text Extraction from Documents*:
   - Extracts text from different types of documents using various libraries like pdfplumber, pytesseract, openpyxl, etc.

2. *Text Processing*:
   - Splits extracted text into smaller chunks for better processing.

3. *Embedding and Vectorization*:
   - Utilizes embedding models like Google Generative AI Embeddings to convert text chunks into vector representations.
   - Uses FAISS library for efficient vector indexing and similarity search.

4. *Conversational AI Setup*:
   - Configures a conversational AI system using models like Google Generative AI to generate responses to user queries.
   - Implements retrieval augmented generation for contextual learning, enhancing the AI's ability to provide contextually relevant responses.

5. *User Interaction*:
   - Provides a web interface for users to ask questions and receive responses from the conversational AI system.
   - Supports uploading various document types for processing.

6. *Document Processing*:
   - Processes uploaded documents to extract text and feed it into the conversational AI system.
   - Supports PDFs, PowerPoint presentations, DOCX files, TXT files, LaTeX files, CSV files, Excel files, and PNG images.

7. *Integration with Conversational Chain*:
   - Integrates the conversational AI system with a conversational retrieval chain to handle user queries effectively.

8. *User Input Handling*:
   - Handles user input questions and provides responses using the configured conversational chain.

9. *Web Interface*:
    - Provides a user-friendly web interface using Streamlit for easy interaction with SciBot.

## Dependencies and Installation
----------------------------
To install the SciBot, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```commandline
OPENAI_API_KEY=your_secret_api_key
```
4. Install Python version 3.9.5
  
   You can download the Python 3.9.5 installer for Windows from the official Python website.
   
## Usage
-----
To use the SciBot, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run main.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple documents into the SciBot by following the provided instructions.

5. Ask questions in natural language about the loaded documents using the chat interface.

## Contributing
------------
This repository is intended for educational purposes. It is created for Mined Hackathon 2024.
