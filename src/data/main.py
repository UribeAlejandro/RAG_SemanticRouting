from src.data.etl import etl_pipeline
from src.data.utils import embeddings_model

if __name__ == "__main__":
    embedding_function = embeddings_model()
    etl_pipeline(embedding_function)
