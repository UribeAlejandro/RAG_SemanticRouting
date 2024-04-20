from dotenv import load_dotenv

from src.pipeline.model import get_pipeline

load_dotenv()

if __name__ == "__main__":
    chain = get_pipeline()
    print(chain.invoke("What is the capital of France?"))
