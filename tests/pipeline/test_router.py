import pytest
from src.pipeline.router import question_router


@pytest.mark.skip(reason="Requires installing Ollama & downloading in GHA")
def test_question_router():
    question = "llm agent memory"
    q_router = question_router()
    assert q_router.invoke({"question": question}) == {"datasource": "vectorstore"}