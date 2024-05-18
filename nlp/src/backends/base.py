from abc import ABC
from typing import List, Union

class NLPInferenceBackend(ABC):
    def __init__(self, **kwargs):
        # Initialize inference backend
        pass

    def infer(
            self,
            inputs: Union[List[List[dict]], List[str]],
            batch_size: int = 128,
            **kwargs
    ) -> List[str]:
        return []