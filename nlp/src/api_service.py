from fastapi import FastAPI, Request

from src.config import get_config
from src.NLPManager import NLPManager
from src.schema import InferenceRequest


app = FastAPI()

config = get_config()
nlp_manager = NLPManager(config)


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/extract")
async def extract(instance: InferenceRequest):
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
    batch_size = config.batch_size
    transcripts = [
        transcript.transcript for transcript in instance.instances
    ]
    predictions = [pred.model_dump() for pred in nlp_manager.qa(
        transcripts, batch_size=batch_size
    )]

    # TODO: serialize with orjson instead of default used by fastapi to squeeze out extra speed
    return {"predictions": predictions}
