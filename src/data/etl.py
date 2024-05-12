from langchain import embeddings as LangchainEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from src import logger
from src.constants import DATA_URLS
from src.data.utils import embeddings_model, vector_database


def extract(urls: list[str] = DATA_URLS) -> list[list[Document]]:
    """Extracts documents from the web.

    Parameters
    ----------
    urls: List[str]
        List of URLs to extract documents from

    Returns
    -------
    List[List[Document]]
        List of lists of documents
    """
    logger.info("Extracting documents")
    docs = [WebBaseLoader(url).load() for url in urls]
    return docs


def transform(docs: list[list[Document]]) -> list[Document]:
    """Transforms the documents into a list of documents.

    Parameters
    ----------
    docs: List[List[Document]]
        List of lists of documents

    Returns
    -------
    List[Document]
        List of documents
    """
    logger.info("Transforming documents")
    docs_list = [item for sublist in docs for item in sublist]

    embeddings = embeddings_model()
    text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
    doc_splits = text_splitter.split_documents(docs_list)

    return doc_splits


def load(vectorstore: Chroma, doc_splits: list[Document]) -> None:
    """Loads the documents into the vectorstore.

    Parameters
    ----------
    vectorstore: Chroma
        The vectorstore to load the documents into
    doc_splits: List[Document]
        The list of documents to load into the vectorstore
    """
    logger.info("Loading documents into vector store")
    vectorstore.add_documents(doc_splits)


def etl_pipeline(embedding_function: LangchainEmbeddings) -> None:
    """ETL pipeline.

    Parameters
    ----------
    embedding_function: LangchainEmbeddings
        The embedding function to use
    """
    logger.info("Starting ETL pipeline")
    docs = extract()
    doc_splits = transform(docs)
    vector_store = vector_database(embedding_function)
    load(vector_store, doc_splits)
