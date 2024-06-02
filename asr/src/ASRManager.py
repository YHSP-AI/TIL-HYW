import io
from faster_whisper import WhisperModel


class ASRManager:
    def __init__(self, whisper_model_name="whisper-til-ct2", max_length=225):
        self.whisper_model = WhisperModel(whisper_model_name)
        self.supressed_tokens = self.get_suppressed_tokens(
            self.whisper_model.hf_tokenizer
        )
        self.max_length = max_length

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        audio_buffer = io.BytesIO(audio_bytes)
        transcripts, _ = self.whisper_model.transcribe(
            audio_buffer,
            "en",
            max_new_tokens=self.max_length,
            suppress_tokens=self.supressed_tokens,
        )
        return " ".join(transcript.text for transcript in transcripts)

    def get_suppressed_tokens(self, tokenizer):
        # tokens_file_path = Path("./cache_tokens")
        bad_tokens_str = [
            r"0",
            r"1",
            r"2",
            r"3",
            r"4",
            r"5",
            r"6",
            r"7",
            r"8",
            r"9",
            # "$",
            # "£",
        ]

        tokens_dict = dict()

        all_tokens_cached = True

        for token_num in range(tokenizer.get_vocab_size()):
            token_str = tokenizer.decode([token_num]).removeprefix(" ")
            for bad_token in bad_tokens_str:
                if bad_token not in tokens_dict:
                    tokens_dict[bad_token] = [False, [], []]
                    all_tokens_cached = False

                if not tokens_dict[bad_token][0]:
                    if bad_token in token_str:
                        tokens_dict[bad_token][1].append(token_num)
                        tokens_dict[bad_token][2].append(token_str)
            if all_tokens_cached:
                break

        result = []
        for k, v in tokens_dict.items():
            tokens_dict[k][0] = True
            if k in bad_tokens_str:
                result += v[1]
        return list(set(result))
