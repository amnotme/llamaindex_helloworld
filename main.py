import os
from dotenv import load_dotenv
import llama_index as llama


def main(url: str) -> None:
    documents = llama.readers.SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = llama.VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()
    response = query_engine.query('What is LlamaIndex')
    print(response)

if __name__ == '__main__':
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    main(url='https://medium.com/llamaindex-blog/llamaindex-newsletter-2024-02-06-9a303130ad9f')
