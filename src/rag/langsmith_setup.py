"""
LangSmith Setup and Observability
Production monitoring for LangChain/LangGraph
"""

import os
from urllib import response
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

#Enable tracing
os.environ["LANGSMITH_TRACING"]="true"

@traceable(name="basic_chaining" ,
            tags=["production", "summarization"],
            # Name your runs for easier identification.
             metadata={"category": "metadata", "purpose": "filtering"} )
            # Add metadata to traces for filtering.


def demo_basic_tracing():

    try:
        llm=ChatOpenRouter(
            model="openrouter/free",
            temperature=0.2)



        prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentance but with one rule you have to talk like anime cat girl , for example ending the sentance with a 'meow' ")

        chain =prompt| llm | StrOutputParser()

        print("Basic tracing demo:\n")

        result = chain.invoke({"topic":"machine learing"})

        print(f"Result : {result}")

    except Exception as e:
        print("Error type:", type(e).__name__)
        print("Error details:", repr(e))

if __name__ == "__main__":
    demo_basic_tracing()