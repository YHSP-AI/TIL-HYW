from fastapi import FastAPI, Request

from src.config import get_config
from src.NLPManager import NLPManager


app = FastAPI()

config = get_config()
nlp_manager = NLPManager()


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/extract")
async def extract(instance: Request):
    """
    Performs QA extraction given a context string

    returns a dictionary with fields:

    {
        "heading": str,
        "target": str,
        "tool": str,
    }
    """
    # get transcription, and pass to NLP model
    request_dict = await instance.json()

    predictions = []
    # TODO: set up batching, batch size should be configurable
    for instance in request_dict["instances"]:
        # each is a dict with one key "transcript" and the transcription as a string
        answers = nlp_manager.qa(instance["transcript"])
        predictions.append(answers)

    # TODO: serialize with orjson instead of default used by fastapi to squeeze out extra speed
    return {"predictions": predictions}
