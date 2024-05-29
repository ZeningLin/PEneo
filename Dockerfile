FROM cnstark/pytorch:2.0.0-py3.9.12-cuda11.7.1-ubuntu22.04

COPY ["data", "model", "pipeline", "start", "tools", "README.md", "requirements.txt", "/workspace/"]

RUN cd /workspace \
    && apt update \
    && pip install --upgrade pip \
    && pip install -r requirements.txt