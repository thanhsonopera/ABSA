FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Install Python 3.11.5
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    curl \
    tzdata \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py \
    && apt-get clean

COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY . .

COPY checkpoint/cp6/model.pth checkpoint/cp6/model.pth

COPY sentiment/checkpoint/good/2/model.pth /app/sentiment/checkpoint/good/2/model.pth
COPY sentiment/model_sentiment.py /app/sentiment/model_sentiment.py
COPY sentiment/evaluate.py /app/sentiment/evaluate.py
COPY sentiment/WKPooling.py /app/sentiment/WKPooling.py

EXPOSE 5000

CMD ["python3.11", "app.py"]