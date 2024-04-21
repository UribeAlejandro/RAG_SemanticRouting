# RAG with Semantic Routing

[![Continuous Integration](https://github.com/UribeAlejandro/RAG_SemanticRouting/actions/workflows/ci.yml/badge.svg)](https://github.com/UribeAlejandro/RAG_SemanticRouting/actions/workflows/ci.yml)

This repository contains the code for building and running a RAG system that includes Semantic Routing.

## Setup the environment

### Installation

Create a virtual environment:

```bash
make venv
```

Install the required packages:

```bash
pip install -r requirements-dev.txt
```

### Models

[Ollama](https://ollama.com/download) software is required to serve models, should be installed before running the code and the models should be downloaded. To download them:

```bash
make download-models
```

### Langsmith

To use the [Langsmith](https://smith.langchain.com/) API, the following environment variables need to be set:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=<Project_Name>
LANGCHAIN_ENDPOINT=<LANGCHAIN_ENDPOINT>
LANGCHAIN_API_KEY=<API_KEY>
```

`Langsmith` keeps track of the project's usage, tracing, and failures. An example of a trace can be found in the [link](https://smith.langchain.com/public/88e836f2-43ef-4e5f-a6d9-3362c4fd0e95/r).

### Web Search

The web search is done using [Tavily AI](https://tavily.com/). To use the API, the following environment variables need to be set:

```bash
TAVILY_API_KEY=<API_KEY>
```

## Running the code

### Extract, Transform & Load

Run the ETL pipeline:

```bash
python -m src.main
```

The previous step will create a [Chroma](https://docs.trychroma.com/) vector database with the `database/` folder.

### Query

Run to access the frontend:

```bash
make run-frontend
```
