from dotenv import load_dotenv
import os

if __name__ == '__main__':
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY') or not os.getenv('PINECONE_API_KEY'):
        raise Exception('Not all environment variables have been loaded')
    print('Going to investigation')