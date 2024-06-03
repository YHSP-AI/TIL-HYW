from faster_whisper import WhisperModel

def cache_model(model_name="whisper-til-ct2"):
    whisper_model = WhisperModel(model_name)
    
if __name__ == "__main__":
    cache_model()
