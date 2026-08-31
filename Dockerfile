FROM python:3.11-slim

WORKDIR /app

# Create a non-root user (standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY --chown=user . /app

# Set write permissions for uploads and model artifacts
RUN chmod -R 777 /app

USER user

EXPOSE 7860

CMD ["python", "app.py"]
