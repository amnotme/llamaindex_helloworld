# LlamaIndex Document Helper Chat Application

This is a chat application powered by LlamaIndex, a Python library for building applications using large language models (LLMs). The application allows users to ask questions about LlamaIndex and receive responses from the assistant.

## Prerequisites

- Python installed
- Basic understanding of Python programming

## Installation


1. **Clone the Project Repository**
   - Use Git to clone the repository to your local machine.

2. **Install Dependencies with Pipenv**
   - In the project's root directory, run the following command to set up a virtual environment with Python 3.10 and install the required packages:
     ```bash
     pipenv --python 3.10 install
     ```

3. **Set Up Environment Variables**
   - Populate the `.env` file with your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     PINECONE_API_KEY=your_pinecone_api_key
     PINECONE_INDEX_HOST=your_pinecone_index_host
     ```

## Usage

Follow these steps to run the LlamaIndex Document Helper:

1. **Activate the Virtual Environment**
   - Before running the script, activate the pipenv shell to ensure you're using the project's virtual environment:
     ```bash
     pipenv shell
     ```

2. **Run the Script**
   - Start the main script via the command line:
     ```bash
     streamlit run main.py
     ```
   - Open your web browser and navigate to http://localhost:8501 to access the chat interface. 
   - Ask questions about LlamaIndex and interact with the assistant.


## Features
- Chat with the LlamaIndex assistant to get answers about LlamaIndex.
- Relevant responses powered by LlamaIndex's vector index and chat engine.
- References and sources provided by the assistant for further reading.