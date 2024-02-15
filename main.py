import os
import nltk

nltk_data_dir = "./resources/nltk_data_dir/"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

import openai
from time import sleep
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.settings import Settings
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llamaindex_docs_chat_app.node_postprocessors.duplicate_postprocessing import (
    DuplicateRemoverNodePostProcessor,
)
from llamaindex_docs_chat_app.llamaindex_docs_constants.llamaindex_docs_file_references import (
    CITATION_REFERENCES,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import streamlit as st
import pinecone


load_dotenv()


def set_page_config() -> None:
    """
    Sets up the page configuration for the web app.

    Args:
        st: Streamlit object.
    """
    st.set_page_config(
        page_title="Chat with Llama, Powered by LlamaIndex",
        page_icon="🦙",
        menu_items=None,
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.title("Chat with LlamaIndex docs v0.10.3, 💬🦙")


def set_sidebar() -> None:
    """
    Sets up the sidebar content.

    Args:
        st: Streamlit object.
    """
    with st.sidebar:
        st.header("💻  About the Llamadocs Chat App")
        st.write(
            """
            This is a chat interface powered by LlamaIndex, designed to provide responses to user queries regarding LlamaIndex. 
            
            Highlights:
            
            Interactive Interface: Engage with an AI for insights on LlamaIndex.
            Advanced AI: Utilizes LlamaIndex and Retrieval-Augmented Generation (RAG) for accurate responses.
            Easy Setup: Straightforward installation and user-friendly instructions.
            Resourceful Sidebar: Quick access to developer info and professional links.
            Innovations:
            
            Pinecone Integration: Enhances response accuracy with vector similarity search.
            Optimized Chat Engine: Ensures relevance and uniqueness in responses.
            """
         )
        st.divider()

        st.header("👨‍💻  About The Author")
        st.write(
            """
            Leo is a senior software engineer based in South Houston, TX, with a background in computer science from WGU. Working in tech, Leo is dedicated to contributing to innovative projects while pursuing the goal of starting his own business in the tech industry. Passionate about programming and technology trends, Leo enjoys engaging with the tech community to share insights and discuss code.

            Connect and say hi!
            """
        )

        st.divider()
        st.subheader("🔗 Connect with Me", anchor=False)
        st.markdown(
            """
            - [🐙 Source Code](https://github.com/amnotme/streamlit_llamadocs_chat)
            - [👔 LinkedIn](https://www.linkedin.com/in/leopoldo-hernandez/)
            """
        )

        st.divider()
        st.write("Made with ♥, powered by LlamaIndex and OpenAI")


@st.cache_resource(show_spinner=True)
def get_index(api_key) -> VectorStoreIndex:
    """
    Retrieves the vector index.

    Returns:
        VectorStoreIndex: The vector index.
    """
    index_name = "llamaindex-documentation-helper"

    pc_index = pinecone.Pinecone().Index(
        name=index_name, host=os.getenv("PINECONE_INDEX_HOST")
    )
    vector_store = PineconeVectorStore(pinecone_index=pc_index)

    Settings.callback_manager = CallbackManager(
        handlers=[LlamaDebugHandler(print_trace_on_end=True)]
    )
    Settings.embed_model = OpenAIEmbedding(api_key=api_key)
    Settings.llm = OpenAI(api_key=api_key)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def retrieve_augmented_generation_response(index):
    """
    Retrieves the response from the chat engine.

    Args:
        st: Streamlit object.
        index: VectorStoreIndex object.
    """
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=Settings.embed_model, percentile_cutoff=0.5, threshold_cutoff=0.7
    )
    if not st.session_state.get("chat_engine"):
        duplicate_node_remover = DuplicateRemoverNodePostProcessor()
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            verbose=True,
            node_postprocessors=[postprocessor, duplicate_node_remover],
        )


def initialize_chat_messages():
    """
    Initializes chat messages.

    Args:
        st: Streamlit object.
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Ask me a question about LlamaIndex open source python Library",
            }
        ]


def get_user_prompt():
    """
    Gets the user input prompt.

    Args:
        st: Streamlit object.

    Returns:
        str: User input prompt.
    """
    if prompt := st.chat_input("What would you like to know about LlamaIndex?"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
    return prompt


def display_messages_on_feed():
    """
    Displays messages on the feed.

    Args:
        st: Streamlit object.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("references"):
                with st.expander("Sources"):
                    for idx, ref in enumerate(message["references"]):
                        st.write(ref)
                        if idx != len(message["references"]) - 1:
                            st.divider()


def reset_state():
    """Initializes necessary states for the app."""
    if "submitted" in st.session_state.keys():
        del st.session_state["submitted"]
    if "openai_key" in st.session_state.keys():
        del st.session_state["openai_key"]
    if "messages" in st.session_state.keys():
        del st.session_state["messages"]
    if "chat_engine" in st.session_state.keys():
        del st.session_state["chat_engine"]
    openai.api_key = None


def update_api_key(api_key):
    """Updates the api key in case it is incorrect"""
    openai.api_key = api_key
    Settings.llm = OpenAI(api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(api_key=api_key)


def store_messages_with_references(prompt):
    """
    Stores messages with references.

    Args:
        st: Streamlit object.
    """
    response = None  # Initialize response to None
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                try:
                    response = st.session_state.chat_engine.stream_chat(message=prompt)
                except Exception as e:
                    if "401" in str(e):
                        st.error(
                            "Incorrect API key provided. Please check and update your API key. The app will now reload!."
                        )
                        reset_state()
                        sleep(5)
                        st.rerun()
                    else:
                        st.write("Sorry, I wasn't able to get any info on that!")
                        st.error(f"An error occurred: {e}")
                        return  # Exit the function early

                if response:
                    st.write_stream(response.response_gen)
                    nodes = [node for node in response.source_nodes]
                    references = []
                    with st.expander("Sources"):
                        with st.spinner("Loading citations"):
                            for idx, node in enumerate(nodes):
                                file_path = CITATION_REFERENCES.get(
                                    node.metadata.get("file_name")
                                )
                                if not file_path:
                                    file_path = f"https://docs.llamaindex.ai/en/stable/ - Look for {node.metadata.get('file_name')}"
                                reference = (
                                    f"📖 Reference: {idx + 1} - 📊 Score - % {round(node.score * 100, 2)} \n"
                                    f"🔗  {file_path}"
                                )
                                references.append(reference)
                                st.write(reference)
                                if idx != len(nodes) - 1:
                                    st.divider()
                    stored_response = {
                        "role": "assistant",
                        "content": response.response,
                        "references": [],
                    }
                    if references:
                        stored_response["references"] = references
                    st.session_state.messages.append(stored_response)


def initialize_state_manager():
    """Initializes necessary states for the app."""
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = None


def get_open_api_key():
    """Manages OpenAI API key input and submission logic."""
    if not st.session_state.submitted:
        with st.form("user_input", clear_on_submit=True):
            openai_api_key = st.text_input(
                "Enter your OpenAI API Key:", placeholder="sk-XXXX", type="password"
            )
            submitted = st.form_submit_button("Ready to chat with LlamaIndex")
            if submitted:
                if openai_api_key:
                    update_api_key(api_key=openai_api_key)
                    st.session_state.submitted = True
                    st.session_state.openai_key = openai_api_key
                    st.rerun()
                else:
                    st.warning("Please enter your OpenAI API key to proceed.")


def main_chat_functionality():

    if st.session_state.openai_key:  # Ensure API key is set
        index = get_index(api_key=st.session_state.openai_key)

        initialize_chat_messages()

        retrieve_augmented_generation_response(index=index)

        prompt = get_user_prompt()

        display_messages_on_feed()

        if prompt:
            store_messages_with_references(prompt=prompt)


if __name__ == "__main__":
    set_page_config()

    set_sidebar()

    initialize_state_manager()

    get_open_api_key()

    if st.session_state.submitted:
        main_chat_functionality()
