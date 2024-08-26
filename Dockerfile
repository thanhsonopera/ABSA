FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Install Python 3.11 and Java 
RUN apt-get update --fix-missing && apt-get install -y git && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update --fix-missing && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    curl \
    tzdata \
    openjdk-17-jdk \
    build-essential \ 
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py \
    && python3.11 -m pip install tensorflow \
    && apt-get clean


COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt

RUN pip3 install torch torchvision torchaudio

COPY . .

COPY checkpoint/cp6/model.pth checkpoint/cp6/model.pth

# Sentiment model
COPY sentiment/checkpoint/good/9/model.pth /app/sentiment/checkpoint/good/9/model.pth
COPY sentiment/model_sentiment.py /app/sentiment/model_sentiment.py
COPY sentiment/evaluate.py /app/sentiment/evaluate.py
COPY sentiment/WKPooling.py /app/sentiment/WKPooling.py

# Install fairseq
COPY fairseq-0.12.3.1-cp311-cp311-linux_x86_64.whl /app/
RUN pip install /app/fairseq-0.12.3.1-cp311-cp311-linux_x86_64.whl
RUN pip show fairseq

# Clone and install fastBPE
RUN git clone https://github.com/glample/fastBPE.git && \
    cd fastBPE && \
    pip install .
RUN pip show fastBPE

EXPOSE 5000

CMD ["python3.11", "app.py"]