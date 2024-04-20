from langchain_core.vectorstores import VectorStoreRetriever

from src.data.utils import embeddings_model, vector_database
import pytest
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(raise_error_if_not_found=True))


@pytest.fixture(scope="module")
def retriever() -> VectorStoreRetriever:
    embeddings = embeddings_model()
    chroma = vector_database(embeddings)
    chroma_retriever = chroma.as_retriever()

    yield chroma_retriever
