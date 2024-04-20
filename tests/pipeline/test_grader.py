import pytest
from src.pipeline.grader import retrieval_grader, hallucination_grader
from src.pipeline.rag import generate_answer


@pytest.mark.skip(reason="Requires installing Ollama & downloading in GHA")
def test_retrieval_grader_docs(retriever):
    ret_grader = retrieval_grader()
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    assert ret_grader.invoke({"question": question, "document": doc_txt}) == {"score": "yes"}


@pytest.mark.skip(reason="Requires installing Ollama & downloading in GHA")
def test_retrieval_grader_text(retriever):
    ret_grader = retrieval_grader()
    question = "agent memory"
    doc_txt = "This is a test document that should not be relevant to the question. It is a test document."
    assert ret_grader.invoke({"question": question, "document": doc_txt}) == {"score": "no"}


@pytest.mark.skip(reason="Requires installing Ollama & downloading in GHA")
def test_hallucination_grader_text(retriever):
    question = "agent memory"
    generation = "Abraham Lincoln was the first president of the United States. He was born in 1809."

    docs = retriever.invoke(question)
    hal_grader = hallucination_grader()
    hal_grader.invoke({"documents": docs, "generation": generation})

    assert hal_grader.invoke({"documents": docs, "generation": generation}) == {"score": "no"}


@pytest.mark.skip(reason="Requires installing Ollama & downloading in GHA")
def test_hallucination_grader_docs(retriever):
    question = "agent memory"
    rag = generate_answer()
    generation = rag.invoke({"question": question, "context": retriever.invoke(question)})

    docs = retriever.invoke(question)
    hal_grader = hallucination_grader()
    hal_grader.invoke({"documents": docs, "generation": generation})

    assert hal_grader.invoke({"documents": docs, "generation": generation}) == {"score": "yes"}
