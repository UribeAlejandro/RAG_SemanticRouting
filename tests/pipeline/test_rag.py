import pytest
from src.pipeline.rag import generate_answer


@pytest.mark.skip(reason="Requires installing Ollama & downloading in GHA")
def test_generate_answer(retriever):
    rag_chain = generate_answer()

    question = "agent memory"
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    assert generation
