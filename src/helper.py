from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os



# Extract text from PDF files
def load_pdf_files(data):
  try:
    # Check if the data directory exists
    if not os.path.exists(data):
      raise ValueError(f"Data directory {data} does not exist.")
    else:
      loader = DirectoryLoader(
          # Data Location
          data,
          glob="*.pdf",
          loader_cls=PyPDFLoader
      )
      documents = loader.load()
      return documents
  except Exception as e:
    print(f"Error loading PDF files: {e}")
    return []


# Filter documents to retain only minimal metadata
def filter_to_minimal_docs(docs):
  """
  Given a list of Document objects, return a new list of Document objects
  containing only 'source' in metadata and the original page_content.
  """
  minimal_docs = []

  for doc in docs:
    src =  doc.metadata.get('source')
    minimal_docs.append(
      Document(
        page_content = doc.page_content,
        metadata = {'source' : src}
      )
    )
  
  return minimal_docs



# Split the documents into smaller chunks 
def text_split(minimal_docs):
  """
  Split the documents into smaller chunks using RecursiveCharacterTextSplitter.
  """
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, 
    chunk_overlap = 20, 
  )
  text_chunks = text_splitter.split_documents(minimal_docs)
  return text_chunks



# Download the embedding model
def download_embeddings():
  """
  Download and return the HuggingFace emebddings model.
  """
  try: 
    model_name = "sentence-transformers/all-MiniLM-L6-v1"
    embeddings = HuggingFaceEmbeddings(
      model_name = model_name
    )
    return embeddings
  except Exception as e:
    print(f"Error downloading embeddings: {e}")
    return None