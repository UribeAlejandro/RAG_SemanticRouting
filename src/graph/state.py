from typing import List, Union

from langchain_core.documents import Document
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """Represents the state of the graph.

    Attributes:
        question: str
        generation: str
        web_search: str
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: Union[List[str], List[Document]]
