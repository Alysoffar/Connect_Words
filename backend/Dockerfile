FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/backend

WORKDIR /workspace

COPY backend/requirements.txt /tmp/backend-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/backend-requirements.txt

COPY backend/app /workspace/backend/app
COPY shared /workspace/shared
COPY data /workspace/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
