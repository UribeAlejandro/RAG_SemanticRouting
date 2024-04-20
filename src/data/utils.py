from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

from src import logger
from src.constants import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY


def vector_database(embedding_function: GPT4AllEmbeddings) -> Chroma:
    """Creates a vector store.

    Parameters
    ----------
    embedding_function: GPT4AllEmbeddings
        The embedding function to use

    Returns
    -------
    Chroma
        The vector store
    """
    logger.info("Creating vector store")
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embedding_function,
    )
    return vector_store


def embeddings_model() -> GPT4AllEmbeddings:
    """Creates a GPT4AllEmbeddings model.

    Returns
    -------
    GPT4AllEmbeddings
        The model
    """
    logger.info("Creating embeddings model")
    model = GPT4AllEmbeddings()
    return model
