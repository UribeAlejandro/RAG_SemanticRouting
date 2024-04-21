from dotenv import load_dotenv

from src import logger
from src.graph.workflow import create_workflow

load_dotenv()

if __name__ == "__main__":
    app = create_workflow()

    inputs = {"question": "Who are the Bears expected to draft first in the NFL draft?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            logger.info("Finished running: %s:", key)
    logger.info("%s", value["generation"])
