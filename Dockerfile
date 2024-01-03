# Dockerfile
FROM python:3.8
WORKDIR /app
COPY ml_model.py /app/ml_model.py
CMD ["python", "ml_model.py"]
