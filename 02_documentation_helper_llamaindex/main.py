from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
import os
import pinecone
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.settings import Settings

load_dotenv()


if __name__ == "__main__":
    INDEX_NAME: str = "llamaindex-documentation-helper"

    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pc_index = pc.Index(name=INDEX_NAME, host=os.getenv("PINECONE_INDEX_HOST"))
    vector_store = PineconeVectorStore(pinecone_index=pc_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    Settings.callback_manager = callback_manager

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query = "What is a LlamaIndex query engine"

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)
