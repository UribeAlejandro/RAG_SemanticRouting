from langchain.llms.gpt4all import GPT4All
from langchain_core.prompts import PromptTemplate


def get_pipeline():
    template = """
    Question: {question}
    Answer: Let's think step by step.
    """
    prompt = PromptTemplate.from_template(template)

    llm = GPT4All(model="./models/mistral-7b-openorca.gguf2.Q4_0.gguf")
    chain = prompt | llm

    return chain
