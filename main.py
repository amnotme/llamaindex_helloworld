from dotenv import load_dotenv, set_key
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
import streamlit as st
import pinecone
import os

load_dotenv()


def set_page_config(st) -> None:
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
    st.title("Chat with LlamaIndex docs, 💬🦙")


def set_sidebar(st) -> None:
    """
    Sets up the sidebar content.

    Args:
        st: Streamlit object.
    """
    with st.sidebar:
        st.header("👨‍💻 About Leo")
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
            - [🐙 GitHub](https://github.com/amnotme)
            - [👔 LinkedIn](https://www.linkedin.com/in/leopoldo-hernandez/)
            """
        )

        st.divider()
        st.write("Made with ♥, powered by LlamaIndex and OpenAI")


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    """
    Retrieves the vector index.

    Returns:
        VectorStoreIndex: The vector index.
    """
    index_name = "llamaindex-documentation-helper"

    pc = pinecone.Pinecone()
    pc_index = pc.Index(name=index_name, host=os.getenv("PINECONE_INDEX_HOST"))
    vector_store = PineconeVectorStore(pinecone_index=pc_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    Settings.callback_manager = callback_manager
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def retrieve_augmented_generation_response(st, index):
    """
    Retrieves the response from the chat engine.

    Args:
        st: Streamlit object.
        index: VectorStoreIndex object.
    """
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=Settings.embed_model, percentile_cutoff=0.5, threshold_cutoff=0.8
    )
    if not st.session_state.get("chat_engine"):
        duplicate_node_remover = DuplicateRemoverNodePostProcessor()
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            verbose=True,
            node_postprocessors=[postprocessor, duplicate_node_remover],
        )


def initialize_chat_messages(st):
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


def get_user_prompt(st):
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


def display_messages_on_feed(st):
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


def store_messages_with_references(st):
    """
    Stores messages with references.

    Args:
        st: Streamlit object.
    """
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                try:
                    response = st.session_state.chat_engine.stream_chat(message=prompt)
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
                except Exception as e:
                    st.write("Sorry, I wasn't able to get any info on that!")


def get_open_api_key(st):

    with st.form("user_input"):
        OPENAI_API_KEY = st.text_input(
            "Enter your OpenAI API Key:", placeholder="sk-XXXX", type="password"
        )
        submitted = st.form_submit_button("Ready to chat with LlamaIndex")

    if submitted:
        if not OPENAI_API_KEY:
            st.info(
                "Please fill out the OpenAI API Key to proceed. If you don't have one, you can obtain it [here]("
                "https://platform.openai.com/account/api-keys)."
            )
            st.stop()
        else:
            set_key(
                dotenv_path=".env",
                key_to_set="OPENAI_API_KEY",
                value_to_set=OPENAI_API_KEY,
            )


if __name__ == "__main__":
    set_page_config(st=st)

    set_sidebar(st=st)

    if not os.getenv('OPENAI_API_KEY'):
        get_open_api_key(st=st)
    else:
        index = get_index()

        initialize_chat_messages(st=st)

        retrieve_augmented_generation_response(st=st, index=index)

        prompt = get_user_prompt(st=st)

        display_messages_on_feed(st=st)

        store_messages_with_references(st=st)
