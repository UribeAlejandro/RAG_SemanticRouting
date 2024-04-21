from typing import Dict, List, Union

from langchain.schema import Document

from src import logger
from src.data.utils import vector_database
from src.graph.state import GraphState
from src.pipeline.grader import retrieval_grader
from src.pipeline.rag import generate_answer
from src.tools.search import search_tool


def retrieve(state: GraphState) -> Dict[str, Union[List[Document], List[str]]]:
    """Retrieve documents from vectorstore.

    Parameters
    ----------
    state : GraphState
        The current graph state
    Returns
    -------
    Dict[str, Union[List[Document], List[str]]]
        New key added to state, documents, that contains retrieved documents
    """
    logger.info("Retrieving documents")

    question = state["question"]

    chroma_db = vector_database()
    retriever = chroma_db.as_retriever()
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}


def generate(state: GraphState) -> Dict[str, Union[List[Document], List[str]]]:
    """Generate answer using RAG on retrieved documents.

    Parameters
    ----------
    state : GraphState
        The current graph state
    Returns
    -------
    Dict[str, Union[List[Document], List[str]]]
        New key added to state, generation, that contains LLM generation
    """
    logger.info("Generating answer")

    question = state["question"]
    documents = state["documents"]

    rag_chain = generate_answer()
    generation = rag_chain.invoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state: GraphState) -> Dict[str, Union[List[Document], List[str]]]:
    """Grade documents based on relevance to question.

    Parameters
    ----------
    state : GraphState
        The current graph state
    Returns
    -------
    Dict[str, Union[List[Document], List[str]]]
        Filtered out irrelevant documents and updated web_search state
    """
    logger.info("Grading documents")

    filtered_docs = []
    should_web_search = "no"
    question = state["question"]
    documents = state["documents"]

    ret_grader = retrieval_grader()

    for d in documents:
        score = ret_grader.invoke({"question": question, "document": d.page_content})
        grade = score["score"]

        if grade.lower() == "yes":
            logger.info("Document relevant")
            filtered_docs.append(d)
        else:
            logger.info("Document not relevant")
            should_web_search = "yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": should_web_search}


def web_search(state: GraphState) -> Dict[str, Union[List[Document], str]]:
    """Web search based on the question.

    Parameters
    ----------
    state : GraphState
        The current graph state
    Returns
    -------
    Dict[str, Union[List[Document], str]]
        Appended web results to documents
    """
    logger.info("Web search")

    question = state["question"]
    documents = state["documents"]

    web_search_tool = search_tool()
    docs = web_search_tool.invoke({"query": question})
    web_results_text = "\n".join([d["content"] for d in docs])
    web_results_doc = Document(page_content=web_results_text)

    if documents is not None:
        documents.append(web_results_doc)
    else:
        documents = [web_results_doc]
    return {"documents": documents, "question": question}
