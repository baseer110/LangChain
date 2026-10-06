from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch


load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that searches for information about a given query. 
    Args:
        query (str): The search query.
    Returns:
        str: A string containing the search results.
    """
    print(f"Searching for: {query}")
    response = tavily_client.search(query=query)
    return response["results"][0]["content"] if response["results"] else "No results found."

llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
   print("Hello from React Search Agent")
   result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})
   print(result)



if __name__ == "__main__":
    main()


def create_agent():
     print("Hello from langchain-course!")
     information = """Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman, industrialist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of September 2026, Forbes estimates his net worth to be US$964 billion."""

     summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about the person
    """

     summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
     )

     # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
     llm = ChatOllama(model="gemma3:270m", temperature=0)
    
     chain = summary_prompt_template | llm
     response = chain.invoke(input={"information": information})
     
     print(response.content)