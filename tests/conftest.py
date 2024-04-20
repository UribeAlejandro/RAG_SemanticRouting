import pytest
from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStoreRetriever

from src.data.utils import embeddings_model, vector_database

load_dotenv()


@pytest.fixture(scope="module")
def retriever() -> VectorStoreRetriever:
    embeddings = embeddings_model()
    chroma = vector_database(embeddings)
    chroma_retriever = chroma.as_retriever()

    yield chroma_retriever
