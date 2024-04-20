# RAG with Semantic Routing

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

### Model

The model corresponds to a open-source LLM from [ChatGPT4All](https://gpt4all.io/index.html), to download the large language model:

```bash
make download-model
```

### Langsmith

To use the [Langsmith](https://smith.langchain.com/) API, the following environment variables need to be set:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=<Project_Name>
LANGCHAIN_ENDPOINT=<LANGCHAIN_ENDPOINT>
LANGCHAIN_API_KEY=<API_KEY>
```

## Running the code

To run the code, you can use the following command:

```bash
python -m src.main
```
