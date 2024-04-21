import time
from functools import lru_cache

from langgraph.graph.graph import CompiledGraph

from src.graph.workflow import create_workflow


def stream_response(text: str) -> str:
    """Stream the response to the client.

    Parameters
    ----------
    text : str
        The text to be streamed
    Returns
    -------
    str
        The streamed text
    """
    for word in text.split():
        yield word + " "
        time.sleep(0.05)


@lru_cache()
def get_app() -> CompiledGraph:
    """Get the compiled graph.

    Returns
    -------
    CompiledGraph
        The compiled graph
    """
    return create_workflow()
