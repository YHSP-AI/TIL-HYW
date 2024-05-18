install:
	python -m venv venv && \
	source venv/bin/activate && \
	pip install -r requirements.txt

dev:
	source venv/bin/activate && \
	fastapi dev src/api_service.py --port 5002

run:
	source venv/bin/activate && \
	uvicorn src.api_service:app --port 5002 --host 0.0.0.0


build:
	docker build . -t hyw-nlp

push:
	docker tag hyw-nlp asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-hyw/hyw-asr:latest
	docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-hyw/hyw-asr:latest

submit:
	gcloud ai models upload --region asia-southeast1 --display-name 'hyw-nlp' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-hyw/hyw-nlp:latest --container-health-route /health --container-predict-route /extract --container-ports 5002 --version-aliases default