from dotenv import load_dotenv
import os
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import (
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
        input_dir="./llamaindex_docs",
        file_extractor={".html": UnstructuredReader()},
    )
    documents = directory_reader.load_data(show_progress=True)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(
        model_name="text-embedding-ada-002", embed_batch_size=100
    )
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )

    index_name = "llamaindex-documentation-helper"
    pinecone_index = pc.Index(name=index_name, host=os.getenv("PINECONE_INDEX_HOST"))
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    print("Ingestion finalized")
