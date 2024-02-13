from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.settings import Settings
from llama_index.core.chat_engine.types import ChatMode
import streamlit as st
import pinecone
import os

load_dotenv()


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    INDEX_NAME: str = "llamaindex-documentation-helper"

    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pc_index = pc.Index(name=INDEX_NAME, host=os.getenv("PINECONE_INDEX_HOST"))
    vector_store = PineconeVectorStore(pinecone_index=pc_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    Settings.callback_manager = callback_manager
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


if __name__ == "__main__":
    index = get_index()
    if not st.session_state.get("chat_engine"):
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT, verbose=True
        )

    st.set_page_config(
        page_title="Chat with Llama, Powered by LlamaIndex",
        page_icon="🦙",
        menu_items=None,
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.title("Chat with LlamaIndex docs, 💬🦙")

    if not st.session_state.get("messages"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me a question about LlamaIndex open source python Library",
            }
        ]

    if prompt := st.chat_input("WHat's your question?"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                response = st.session_state.chat_engine.stream_chat(message=prompt)
                st.write_stream(response.response_gen)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.response}
                )
