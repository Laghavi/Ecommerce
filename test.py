from typing import Dict, Any

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
import  json



load_dotenv()

# Set up logging
class CustomHandler(BaseCallbackHandler):
    def __init__(self):
        self.intermediate_steps = []

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self.intermediate_steps.append(outputs)

def callme():


# Initialize Groq LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

    graph = Neo4jGraph(url="neo4j+s://5dc8a1a8.databases.neo4j.io", username="neo4j", password="gh1AN4N1NnTtvrPe0K_OTi7klde0wunK-PRo-1-DmHk")

    chain = GraphCypherQAChain.from_llm(
       llm=llm, graph=graph, verbose=True
    )

    res= chain.invoke({"query": "which product has recipe Hot Chocolate?"})


    custom_handler = CustomHandler()
    print("Default result:", res.get("result"))

    # Print the detailed information from the custom handler
    if custom_handler.intermediate_steps:
        last_step = custom_handler.intermediate_steps[-1]
        if 'intermediate_steps' in last_step:
            for step in last_step['intermediate_steps']:
                if 'context' in step:
                    print("\nDetailed Context:")
                    print(json.dumps(step['context'], indent=2))

if __name__=="__main__":
    callme()