from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.file.unstructured import UnstructuredReader
from llama_index.core.indices.base import IndexType
from llama_index.core.settings import Settings
from typing import Dict, List
import pinecone


def initialize():
    """
    Initialize environment variables.

    Raises:
        Exception: If OPENAI_API_KEY or PINECONE_API_KEY environment variables are missing.
    """
    try:
        load_dotenv()
        if (
            not os.getenv("OPENAI_API_KEY")
            or not os.getenv("PINECONE_API_KEY")
            or not os.getenv("PINECONE_ENVIRONMENT")
        ):
            raise Exception(
                "Not all environment variables have been loaded or are missing."
            )
    except Exception as e:
        print(f"Error occurred while initializing: {str(e)}")


def get_documents_from_directory(
    directory: str, file_extractor_dict: Dict
) -> List[Document]:
    """
    Retrieve documents from a directory using specified file extractors.

    Args:
        directory (str): Path to the directory containing documents.
        file_extractor_dict (Dict): Dictionary mapping file extensions to file extractors.

    Returns:
        List[Document]: List of Document objects extracted from the directory.
    """
    try:
        directory_reader = SimpleDirectoryReader(
            input_dir=directory,
            file_extractor=file_extractor_dict,
        )
        return directory_reader.load_data(show_progress=True)
    except Exception as e:
        print(f"Error occurred while getting documents: {str(e)}")
        return []


def update_llama_index_settings_with_service_context(
    llm: str, openai_model: str
) -> None:
    """
    Update llama index settings with service context.

    Args:
        llm (str): Name of the LLM (language model) to use.
        openai_model (str): Name of the OpenAI model to use for embeddings.
    """
    try:
        node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
        llm = OpenAI(model=llm, temperature=0)
        embed_model = OpenAIEmbedding(model_name=openai_model, embed_batch_size=100)
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
    except Exception as e:
        print(f"Error occurred while updating llama index settings: {str(e)}")


def ingest_documents(index_name: str, documents: List[Document]) -> IndexType:
    """
    Ingest documents into a vector store index.

    Args:
        index_name (str): Name of the index.
        documents (List[Document]): List of Document objects to ingest.

    Returns:
        IndexType: Vector store index containing ingested documents.
    """
    try:
        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pinecone_index = pc.Index(
            name=index_name, host=os.getenv("PINECONE_INDEX_HOST")
        )
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            show_progress=True,
        )
        return index
    except Exception as e:
        print(f"Error occurred while ingesting documents: {str(e)}")
        return None


if __name__ == "__main__":
    # Define constants
    DIRECTORY_TO_READ: str = "./llamaindex_docs_tmp"
    FILE_EXTRACTOR_DICT: Dict = {".html": UnstructuredReader()}
    LLM_MODEL: str = "gpt-3.5-turbo"
    OPENAPI_MODEL: str = "text-embedding-ada-002"
    INDEX_NAME: str = os.getenv("PINECONE_ENVIRONMENT")

    try:
        # Initialize environment variables
        initialize()

        # Retrieve documents from directory
        documents = get_documents_from_directory(
            directory=DIRECTORY_TO_READ, file_extractor_dict=FILE_EXTRACTOR_DICT
        )

        # Update llama index settings with service context
        update_llama_index_settings_with_service_context(
            llm=LLM_MODEL, openai_model=OPENAPI_MODEL
        )

        # Ingest documents into vector store index
        index = ingest_documents(index_name=INDEX_NAME, documents=documents)

        if index:
            print("Ingestion finalized")
        else:
            print("Error occurred during ingestion.")
    except Exception as e:
        print(f"Error occurred in main: {str(e)}")
