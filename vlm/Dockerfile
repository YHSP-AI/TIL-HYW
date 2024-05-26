# example deep learning VM
# for a full list see us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# and for details see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks

# FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py39
FROM hiredscorelabs/pytorch:python3.9.5-1.7.1-cuda10.1-cudnn7-runtime
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


COPY src .

# RUN pip install -r requirements.txt && cd Open-GroundingDino/models/GroundingDINO/ops && python3 setup.py build install --user && cd ../../../ && cd GroundingDINO/ && pip install --no-build-isolation -e . && cd ../
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt && cd GroundingDINO/ && pip install --no-build-isolation -e . && cd ../

# copy the rest of the files into the container

# start model service
CMD uvicorn api_service:app --port 5004 --host 0.0.0.0