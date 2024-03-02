#chatWithSciBot

## Introduction
------------
SciBot is a Python application that allows you to chat with multiple documents in multiple formats. You can ask questions about the documents using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded documents.

## How It Works
------------

![SciBot Architecture Diagram](./architecturelangchain.png)

The application follows these steps to provide responses to your questions:

1. Document Loading: The SciBot reads multiple documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes an OpenAI GPT large language model(LLM) to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the SciBot compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the documents.

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
