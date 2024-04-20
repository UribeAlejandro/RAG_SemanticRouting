from langchain.llms.gpt4all import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable

from src.constants import MODEL_PATH


def get_pipeline() -> RunnableSerializable:
    """Create a pipeline for the RAG model.

    Returns
    -------
    RunnableSerializable
        Langchain Agent
    """
    template = """
    Question: {question}
    Answer: Let's think step by step.
    """
    llm = GPT4All(model=MODEL_PATH)
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    return chain
