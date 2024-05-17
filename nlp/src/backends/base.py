from abc import ABC
from typing import List

class NLPInferenceBackend(ABC):
    def __init__(self, **kwargs):
        # Initialize inference backend
        pass

    def infer(
            self,
            inputs: List[List[dict]],
            batch_size: int = 128,
            **kwargs
    ) -> List[str]:
        return []