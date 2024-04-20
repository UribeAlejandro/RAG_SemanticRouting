from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable

from src.constants import MODEL_NAME


def question_router() -> RunnableSerializable:
    """
    Create a chain for the Question Router.
    Returns
    -------
    RunnableSerializable
        Langchain Agent
    """
    llm = ChatOllama(model=MODEL_NAME, format="json", temperature=0)
    prompt = PromptTemplate(
        template="""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert at routing a user question to a vectorstore or web search. 
        Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. 
        You do not need to be stringent with the keywords in the question related to these topics. 
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. 
        
        Return the a JSON with a single key 'datasource' and no premable or explaination. 
        
        Question to route: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    chain = prompt | llm | JsonOutputParser()
    return chain
