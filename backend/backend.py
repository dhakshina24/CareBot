from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from src.helper import download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time

app = Flask(__name__)
CORS(app)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

# Check if the API keys are set
if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
  raise ValueError("PINECONE_API_KEY and HUGGINGFACE_API_KEY must be set in the .env file.")

# Generate embeddings for the text chunks
embedding = download_embeddings()
print("Embedding Model Downloaded")

index_name = 'medical-chatbot'

# Initialize Pinecone client
pc = Pinecone(api_key = PINECONE_API_KEY)
print("Pinecone Client Initialized")

# Load Existing Index
# Embed each chunk and upsert the embeddings into the index
docsearch = PineconeVectorStore.from_existing_index(
  index_name = index_name, 
  embedding = embedding
)

# Define retriever for retrieving docs from VectorDB
retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={'k': 3})

# Define LLM - Quantized Llama-2
config = {'max_new_tokens': 256, 'context_length': 1024, 'temperature': 0.7, 'threads': 4}
llm = CTransformers(model= "../models/llama-2-7b-chat.Q3_K_S.gguf", model_file="llama-2-7b-chat.Q3_K_S.gguf", model_type="llama", config=config)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

# Define RAG chain and how context should be given to llm
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def home():
  return "Medical Chatbot API running!"

@app.route("/get", methods=["POST"])
def get_response():
  start = time.time()
  try:
    data = request.get_json()
    user_query = data.get("message", "")

    if not user_query:
      return jsonify({"response": "No query provided."}), 400
    
    # Invoke RAG chain
    response = rag_chain.invoke({"input": user_query})

    result = response.get("answer", "")
    return jsonify({"response": result})
  except Exception as e:
    return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__=="__main__":
  app.run(debug=True, port=5000)
