from typing import Dict

from src.backends.llamacpp.backend import LCPPInferenceBackend
from src.backends.base import NLPInferenceBackend
from src.schema import InferenceBackend

class InferenceBackendFactory:
    backends: Dict[InferenceBackend, NLPInferenceBackend] = {
        InferenceBackend.LCPP: LCPPInferenceBackend
    }

    def create(
            self,
            backend: InferenceBackend,
            **kwargs
    ) -> NLPInferenceBackend:
        if backend not in self.backends:
            raise ValueError("Invalid backend selected")
        return self.backends[backend](**kwargs)
    