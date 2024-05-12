from abc import ABC

class NLPInferenceBackend(ABC):
    def __init__(self):
        # Initialize inference backend
        pass

    def infer(
            self,
            inputs: list[str],
            batch_size: int = 128,
            **kwargs
    ) -> list[str]:
        raise NotImplementedError