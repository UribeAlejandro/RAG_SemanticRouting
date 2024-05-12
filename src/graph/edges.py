from src import logger
from src.graph.state import GraphState
from src.pipeline.grader import answer_grader, hallucination_grader
from src.pipeline.router import question_router


def route_question(state: GraphState) -> str:
    """Route question to web search or RAG.

    Parameters
    ----------
    state: GraphState
        The current graph state

    Returns
    -------
    str
        Next node to call
    """
    logger.info("Routing question")
    q_router = question_router()
    question = state["question"]
    source = q_router.invoke({"question": question})

    logger.info("Question to route: %s", question)

    if source["datasource"] == "web_search":
        logger.info("Routing question to web search")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        logger.info("Routing question to RAG")
        return "vectorstore"


def decide_to_generate(state: GraphState) -> str:
    """Determines whether to generate an answer, or add web search.

    Parameters
    ----------
    state: GraphState
        The current graph state

    Returns
    -------
    str
        Binary decision for next node to call
    """
    logger.info("Decide to generate")
    web_search = state["web_search"]

    if web_search == "yes":
        logger.info("All documents are not relevant to question -> include web search")
        return "websearch"
    else:
        logger.info("Documents are relevant to question -> generate answer")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """Determines whether the generation is grounded in the document and
    answers question.

    Parameters
    ----------
    state: GraphState
        The current graph state

    Returns
    -------
    str
        Decision for next node to call
    """
    logger.info("Grade generation vs documents and question")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    ans_grader = answer_grader()
    hal_grader = hallucination_grader()

    logger.info("Checking hallucinations")
    score = hal_grader.invoke({"documents": documents, "generation": generation})
    grade = score["score"]

    if grade == "yes":
        logger.info("Generation is grounded in documents")
        score = ans_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]

        if grade == "yes":
            logger.info("Generation addresses question")
            return "useful"
        else:
            logger.info("Generation does not address question")
            return "not useful"
    else:
        logger.info("Generation is not grounded in documents -> retry")
        return "not supported"
