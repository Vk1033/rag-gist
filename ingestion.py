import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize the OpenAIEmbeddings class configured for OpenRouter
embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    # Crucial: Disables LangChain's string-based token counter since we are passing dictionaries
    check_embedding_ctx_length=False,
    # Passes any extra parameters directly to the underlying OpenAI API request
    model_kwargs={"encoding_format": "float"},
)

# --- Main Logic ---

loader = TextLoader("./mediumblog1.txt")
docs = loader.load()

# Using Recursive splitter to handle text more intelligently
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

try:
    # IMPORTANT: index_name must point to a Pinecone index configured for 2048 dims
    vector_store = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.getenv("INDEX_NAME")
    )
    print("✅ Successfully uploaded chunks to Pinecone!")
except Exception as e:
    print(f"❌ Error: {e}")
