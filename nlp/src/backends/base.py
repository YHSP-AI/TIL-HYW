from abc import ABC

class NLPInferenceBackend(ABC):
    def __init__(self, **kwargs):
        # Initialize inference backend
        pass

    def infer(
            self,
            inputs: list[list[dict]],
            batch_size: int = 128,
            **kwargs
    ) -> list[str]:
        return []