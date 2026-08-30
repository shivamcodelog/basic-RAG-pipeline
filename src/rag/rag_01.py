from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chat_models import init_chat_model


from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel,Field
from typing import List


load_dotenv()
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Sample knowledge base
KNOWLEGDE_BASE= """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

llm= init_chat_model(
    model="minimax/minimax-m2.7:free",
    model_provider="openrouter",
    temperature=0.2
)

def create_kb():
    # split the KNOWLEGDE_BASE into chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    doc= Document(
        page_content=KNOWLEGDE_BASE,
        metadata={"source":"langchain_knowledge_base.md"}
    )

    chunk = splitter.split_documents([doc])

    # create a vector store from chunks

    vector_store=Chroma.from_documents(
        documents=chunk,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    return vector_store

def basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_type= "similarity",search_kwargs={"k":2}
    )

    # RAG prompt template
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:{context}

Question:{question}

Answer:



Make sure to answer in a concise manner,
and if you do not know the answer, just say "I don't know.
"""
    )

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # RAG chain

    rag_chain =(
        {"context":retriever | format_docs , "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Testing RAG chain

    question = [
        "what is langchain",
        "who created langchan",
        "what is lanchain is used for ??"
    ]

    for q in question:
        answer = rag_chain.invoke(q)
        print(f"Q:{q}")
        print(f"A:{answer}")


if __name__ == "__main__":
    basic_rag()
