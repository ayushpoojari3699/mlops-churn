FROM python:3.10-slim

WORKDIR /app

COPY api /app/api
COPY models /app/models
COPY data /app/data

RUN pip install --no-cache-dir fastapi uvicorn joblib pandas scikit-learn

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]