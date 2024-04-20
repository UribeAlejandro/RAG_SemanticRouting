import pytest
from dotenv import load_dotenv

from src.data.utils import embeddings_model, vector_database
from src.pipeline.model import retrieval_grader

load_dotenv()


@pytest.mark.skip(reason="Requires installing Ollama & downloading in GHA")
def test_retrieval_grader():
    embeddings = embeddings_model()
    chroma = vector_database(embeddings)
    retriever = chroma.as_retriever()
    ret_grader = retrieval_grader()

    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    assert ret_grader.invoke({"question": question, "document": doc_txt}).get("score") in ["yes", "no"]
