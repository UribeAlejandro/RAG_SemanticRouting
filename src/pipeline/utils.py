from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """Format a list of documents into a single string.

    Parameters
    ----------
    docs : List[Document]
        List of documents to format.

    Returns
    -------
    str
        Formatted documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)
