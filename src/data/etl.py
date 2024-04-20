from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src import logger
from src.constants import DATA_URLS
from src.data.utils import vector_database


def extract(urls: List[str] = DATA_URLS) -> List[List[Document]]:
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


def transform(docs: List[List[Document]]) -> List[Document]:
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

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)

    return doc_splits


def load(vectorstore: Chroma, doc_splits: List[Document]) -> None:
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


def etl_pipeline(embedding_function: GPT4AllEmbeddings) -> None:
    """ETL pipeline.

    Parameters
    ----------
    embedding_function: GPT4AllEmbeddings
        The embedding function to use
    """
    logger.info("Starting ETL pipeline")
    docs = extract()
    doc_splits = transform(docs)
    vector_store = vector_database(embedding_function)
    load(vector_store, doc_splits)
