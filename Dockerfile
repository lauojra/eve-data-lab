FROM python:3.11-slim

# Install Java for Spark
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

WORKDIR /app

# Copy requirements first for cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install pyspark

# Copy project
COPY . .

CMD ["python", "main.py"]
