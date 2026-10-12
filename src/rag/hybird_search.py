from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain import Document
from typing import List

def hybrid_retriever( query , retrievers , weights , k=3 ,rrf_k=60 ):
    