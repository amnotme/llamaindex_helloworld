from dotenv import load_dotenv
import os
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_hub.file.unstructured import UnstructuredReader
import pinecone


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        raise Exception("Not all environment variables have been loaded")
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    loader = UnstructuredReader()
    directory_reader = SimpleDirectoryReader(
        input_dir="./llamaindex_docs_tmp", file_extractor={".html": UnstructuredReader()}
    )
    documents = directory_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(documents=documents)
    print("Going to investigation")
