from langchain import embeddings as LangchainEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from src import logger
from src.constants import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY, EMBEDDINGS_MODEL


def vector_database(embedding_function: LangchainEmbeddings) -> Chroma:
    """Creates a vector store.

    Parameters
    ----------
    embedding_function: LangchainEmbeddings
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


def embeddings_model() -> LangchainEmbeddings:
    """Creates a GPT4AllEmbeddings model.

    Returns
    -------
    OllamaEmbeddings
        Embeddings model
    """
    logger.info("Creating embeddings model")
    model = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    return model
