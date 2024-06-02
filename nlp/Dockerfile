# example deep learning VM
# for a full list see us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# and for details see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# pip gives a warning if you install packages as root
# set this flag to just ignore the warning
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install -U pip
WORKDIR /workspace

# install other requirements
COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# copy the rest of the files into the container
COPY src src

# Get model
# ADD https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf?download=true src/models/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf
COPY training/output/flan-t5-lora-v1/flan-t5-base-lora-rslora-v1.1 src/models/flan-t5-base-lora-rslora-v1.1

# start model service
CMD uvicorn src.api_service:app --port 5002 --host 0.0.0.0
