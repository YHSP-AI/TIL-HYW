import io
import json
import mmap
import numpy
import soundfile
import torchaudio
import torch

from collections import defaultdict
from pathlib import Path
from pydub import AudioSegment

from seamless_communication.inference import Translator
from seamless_communication.streaming.dataloaders.s2tt import SileroVADSilenceRemover
model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

translator = Translator(
    model_name,
    vocoder_name,
    device=torch.device("cuda:0"),
    dtype=torch.float16,
)
tgt_langs = ("eng")

for tgt_lang in tgt_langs:
  in_file = f"../advanced/audio/audio_0.wav"

  text_output, _ = translator.predict(
      input=in_file,
      task_str="asr",
      tgt_lang=tgt_lang,
  )

  print(f"Transcribed text in {tgt_lang}: {text_output[0]}")