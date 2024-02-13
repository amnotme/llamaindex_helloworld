import os  # Import the os module for working with the operating system
from dotenv import load_dotenv  # Import load_dotenv function from dotenv module
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
def main(url: str) -> None:
    # Read web page content and convert it into text
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents=documents)

    # Create a query engine from the index
    query_engine = index.as_query_engine()

    # Execute a query on the index
    response = query_engine.query("What is LlamaIndex")

    # Print the response
    print(response)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Check if the OPENAI_API_KEY environment variable is set
    if os.getenv("OPENAI_API_KEY") is None:
        # Raise a ValueError if OPENAI_API_KEY is not set
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Call the main function with the specified URL
    main(url="https://www.llamaindex.ai/")
