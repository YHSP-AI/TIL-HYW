#!/bin/sh
ct2-transformers-converter --model src/checkpoint-4000 --output_dir src/whisper-til-ct2 --copy_files preprocessor_config.json --quantization float16