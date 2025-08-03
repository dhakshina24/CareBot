from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score 
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
from transformers import pipeline
import pandas as pd

class RAGEval:
  """"Metric Functions for Evaluation of RAG Pipeline"""
  def __init__(self, candidates="", references=""):
    self.candidates = [candidates]
    self.references = [references]

  def evaluate_bleu(self):
      bleu_score = corpus_bleu(self.candidates, [self.references]).score
      return bleu_score
  
  def evaluate_bert_score(self):
     try: 
       P, R, F1 = score(self.candidates, self.references, lang="en", model_type='bert-base-multilingual-cased')
       return P.mean().item(), R.mean().item(), F1.mean().item()
     
     except RuntimeError as e:
        print('BERTScore could not be computed due to memory constraints.')
  
  def evaluate_rouge(self):
     scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
     rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(self.references, self.candidates)]
     rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
     return rouge1
  

  def evaluate_faithfulness(self):
      nli = pipeline("text-classification", model="facebook/bart-large-mnli")
      input_text = f"premise: {self.references} hypothesis: {self.candidates}"
      faithfulness = nli(input_text)
      return faithfulness[0]["label"], faithfulness[0]["score"]


def build_rag_pipeline():
   """ Function to build RAG Pipeline"""
   load_dotenv()
   PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
   HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
   os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
   os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
   

   # Generate embeddings for the text chunks
  #  embedding_start  = time.time()
   embedding = download_embeddings()
   print("Embedding Model Downloaded")
  #  embedding_end = time.time()
  #  print("Embedding Time:", embedding_end - embedding_start)
   
   
   # Initialize Pinecone client
  #  db_start = time.time()
   pc = Pinecone(api_key = PINECONE_API_KEY)
   # Connect to the index
   index_name = 'medical-chatbot'
   index = pc.Index(index_name)
   print("Pinecone Client Initialized")
   
   # Load Existing Index
   # # Embed each chunk and upsert the embeddings into the index
   index_name = 'medical-chatbot'
   docsearch = PineconeVectorStore.from_existing_index(
      index_name = index_name, 
      embedding = embedding
      )
   
   # Define retriever for retrieving docs from VectorDB
   retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={'k': 3})
  #  db_end = time.time()
  #  print("Pinecone Initialization and Vector DB setup", db_end - db_start)
   
   # Define LLM - Quantized Llama-2
   model_start = time.time()
   config = {'max_new_tokens': 256, 'context_length': 1024, 'temperature': 0.8, 'threads': 4}
   llm = CTransformers(model= "../models/llama-2-7b-chat.Q3_K_S.gguf", model_file="llama-2-7b-chat.Q3_K_S.gguf", model_type="llama", config=config)
  #  model_end = time.time()
  #  print("Time taken to load LLM:", model_end - model_start)
   
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
   return rag_chain


def evaluate_question(question, rag_chain):
  """Evaluate RAG pipeline and return metrics"""
  response_start = time.time()
  rag_response = rag_chain.invoke({'input': question})
  response = rag_response.get("answer", "")
  reference = " ".join([doc.page_content for doc in rag_response['context']])

  # Evaluate RAG
  evaluator = RAGEval(response, reference)
  bleu = evaluator.evaluate_bleu()
  P, R, F1 = evaluator.evaluate_bert_score()
  rouge = evaluator.evaluate_rouge()
  faith_label, faith_score = evaluator.evaluate_faithfulness()
  response_end = time.time()

  return {
     "question": question, 
     "response": response, 
     "context": reference, 
     "BLUE": bleu, 
     "bert_P": P, 
     "bert_R": R, 
     "bert_F1": F1, 
     "rouge1": rouge, 
     "faith_label": faith_label,  
     "faith_score": faith_score,  
     "latency":  response_end - response_start,

  }
   

if __name__ == '__main__':
  start_time = time.time()
  rag_chain = build_rag_pipeline()

  questions = [
        "What is Acne?",
        "What are the symptoms of diabetes?",
        "What causes high blood pressure?",
        "How is asthma diagnosed?",
        "What are the early signs of Alzheimer’s disease?"
  ]

  results = []
  for q in questions: 
     print(f"Evaluating: {q}")
     result = evaluate_question(q, rag_chain)
     results.append(result)
  
  df = pd.DataFrame(results)
  df.to_csv("results.csv", index=False)
  print("Evaluation Completed and Results saved to results.csv")
  end_time = time.time()
  print("Execution Time:", end_time-start_time)