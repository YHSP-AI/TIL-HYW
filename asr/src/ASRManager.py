import io
from faster_whisper import WhisperModel


class ASRManager:
    def __init__(self, whisper_model_name="whisper-til-ct2", max_length=225):
        self.whisper_model = WhisperModel(whisper_model_name, local_files_only=True)
        self.whisper_model.hf_tokenizer.add_prefix_space = False
        self.max_length = max_length

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        audio_buffer = io.BytesIO(audio_bytes)
        transcripts, _ = self.whisper_model.transcribe(
            audio_buffer,
            "en",
            max_new_tokens=self.max_length,
        )
        return " ".join(transcript.text for transcript in transcripts)
