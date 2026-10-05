from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


# def main() -> None:
def main():
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

    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm = ChatOllama(model="gemma3:270m", temperature=0)
    
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)




if __name__ == "__main__":
    main()
