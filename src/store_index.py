import os
from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

# Load PDF files from the directory
extracted_data = load_pdf_files('../data')
print("Extracted Data")

# Filter to minimal documents
filtered_docs = filter_to_minimal_docs(extracted_data)
print("Filtered Documents")

# Split the filtered documents into text chunks
text_chunks = text_split(filtered_docs)
print("Text Chunks Created")

# Generate embeddings for the text chunks
embedding = download_embeddings()
print("Embeddings Downloaded")

# Initialize Pinecone client
pc = Pinecone(api_key = PINECONE_API_KEY)
print("Pinecone Client Initialized")


# Connect to the Vector DB
index_name = 'medical-chatbot'

# Check if the index exists, if not create it
if index_name not in pc.list_indexes():
  pc.create_index(
    name = index_name,
    dimension = 384,
    metric = "cosine",
    spec = ServerlessSpec(
      cloud="aws",
      region = "us-east-1",
    )
  )
  print(f"Index '{index_name}' created successfully.")

  # Create a PineconeVectorStore instance
  docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks, 
    embedding = embedding, 
    index_name = index_name
  )
  print("Pinecone Vector Store created and documents upserted.")

else:
  # Load Existing Index
  # Embed each chunk and upsert the embeddings into the index
  docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name, 
    embedding = embedding
  )