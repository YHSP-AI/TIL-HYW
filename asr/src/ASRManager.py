import numpy as np
import io
import soundfile as sf
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig
import torch
class ASRManager:
    def __init__(self, whisper_model_name="checkpoint-4000/", max_length=225):
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")
        self.generation_config = GenerationConfig.from_pretrained(whisper_model_name)
        self.generation_config.use_cache = True
        self.whisper_processor.tokenizer.add_prefix_space = False

        self.generation_config.suppress_tokens += self.get_suppressed_tokens(self.whisper_processor.tokenizer)
        self.generation_config.suppress_tokens = list(set(self.generation_config.suppress_tokens))
        self.generation_config.suppress_tokens.sort()
        self.generation_config.max_length = max_length

        self.generation_config.max_new_tokens = self.generation_config.max_length
        self.forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(language='english', task='transcribe')
        pass

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        audio_buffer = io.BytesIO(audio_bytes)
        audio_array, sample_rate = sf.read(audio_buffer)

        input_features = self.whisper_processor(audio_array,
                                           sampling_rate=16000,
                                           return_tensors="pt").input_features.to("cuda:0")

        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(input_features,
                                                   generation_config=self.generation_config,
                                                   forced_decoder_ids=self.forced_decoder_ids).to("cuda:0")

        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]


    def get_suppressed_tokens(self, tokenizer):
        # tokens_file_path = Path("./cache_tokens")
        bad_tokens_str = [r"0",
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

        for token_num in range(tokenizer.vocab_size):
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



