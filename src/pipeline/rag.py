from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from src.constants import MODEL_NAME


def generate_answer() -> RunnableSerializable:
    """
    Create a chain for the Answer Generator.
    Returns
    -------
    RunnableSerializable
        Langchain Agent
    """
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    prompt = PromptTemplate(
        template="""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        <|eot_id|><|start_header_id|>user<|end_header_id|>

        Question: {question} 
        Context: {context} 

        Answer: 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )
    chain = prompt | llm | StrOutputParser()

    return chain


