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



## LlamaIndex Chat Interface

This is a chat interface powered by LlamaIndex, designed to provide responses to user queries regarding LlamaIndex. Below is an overview of the functionality and structure of the code:

## Functionality

1. **Importing Necessary Libraries**: The code imports required libraries and modules, including dotenv, llama_index, and streamlit.

2. **Setting Page Configuration**: The `set_page_config` function configures the page layout for the web app using Streamlit.

3. **Setting Sidebar Content**: The `set_sidebar` function sets up the sidebar content, which includes information about the developer and links to GitHub and LinkedIn.

4. **Retrieving Vector Index**: The `get_index` function retrieves the vector index from Pinecone, a vector similarity search service.

5. **Retrieving Response from Chat Engine**: The `retrieve_augmented_generation_response` function retrieves the response from the chat engine, powered by LlamaIndex. It also sets up postprocessors for sentence embedding optimization and duplicate removal.

6. **Initializing Chat Messages**: The `initialize_chat_messages` function initializes the chat messages, setting an initial message from the assistant.

7. **Getting User Input Prompt**: The `get_user_prompt` function obtains the user input prompt from the chat input.

8. **Displaying Chat Messages**: The `display_messages_on_feed` function displays the chat messages on the feed, including any references/sources provided by the assistant.

9. **Storing Messages with References**: The `store_messages_with_references` function stores the user's message and retrieves the assistant's response using the chat engine. It also retrieves and displays any references/sources provided by the assistant.

## Usage

1. **Main Method**: The main method orchestrates the functionality of the chat interface. It starts by retrieving the vector index using the `get_index` function, sets up the page configuration and sidebar, initializes chat messages, retrieves augmented generation responses, obtains user prompts, displays messages on the feed, and stores messages with references.

2. **Interacting with the Interface**: Users can interact with the chat interface by inputting queries about LlamaIndex. The assistant, powered by LlamaIndex and RAG (Retrieval-Augmented Generation), provides relevant and informative responses leveraging the vector index and chat engine provided by LlamaIndex.

## Features
- Chat with the LlamaIndex assistant to get answers about LlamaIndex.
- Relevant responses powered by LlamaIndex's vector index and chat engine.
- References and sources provided by the assistant for further reading.
