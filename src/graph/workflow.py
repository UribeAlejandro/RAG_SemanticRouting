from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from src import logger
from src.graph.edges import decide_to_generate, grade_generation_v_documents_and_question, route_question
from src.graph.nodes import generate, grade_documents, retrieve, web_search
from src.graph.state import GraphState


def create_workflow() -> CompiledGraph:
    """Create a workflow for the Langchain Agent.

    Returns
    -------
    CompiledGraph
        LangGraph Object
    """
    logger.info("Creating workflow")
    workflow = StateGraph(GraphState)

    logger.info("Adding nodes to workflow")
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    logger.info("Adding edges to workflow")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    logger.info("Compiling workflow")
    app = workflow.compile()

    return app
