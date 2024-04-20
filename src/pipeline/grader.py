from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSerializable

from src.constants import MODEL_NAME


def retrieval_grader() -> RunnableSerializable:
    """Create a chain for the Retrieval Grader.

    Returns
    -------
    RunnableSerializable
        Langchain Agent
    """
    llm = ChatOllama(model=MODEL_NAME, format="json", temperature=0)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n

        Answer the binary score as a JSON with a single key 'score' and no preamble or explanation. \n

        <|eot_id|><|start_header_id|>user<|end_header_id|>

        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )
    chain = prompt | llm | JsonOutputParser()

    return chain


def hallucination_grader() -> RunnableSerializable:
    """Create a chain for the Hallucination Grader.

    Returns
    -------
    RunnableSerializable
        Langchain Agent
    """
    llm = ChatOllama(model=MODEL_NAME, format="json", temperature=0)
    prompt = PromptTemplate(
        template="""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded 
        in / supported by a set of facts. 

        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 
        <|eot_id|><|start_header_id|>user<|end_header_id|>

        Here are the facts:
        \n ------- \n
        {documents}
        \n ------- \n
        Here is the answer:
        {generation}  
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["generation", "documents"],
    )
    chain = prompt | llm | JsonOutputParser()

    return chain


def answer_grader() -> RunnableSerializable:
    """Create a chain for the Answer Grader.
    Returns
    -------
    RunnableSerializable
        Langchain Agent
    """
    llm = ChatOllama(model=MODEL_NAME, format="json", temperature=0)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader assessing whether an answer is useful to resolve a question. 
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["generation", "question"],
    )

    chain = prompt | llm | JsonOutputParser()

    return chain