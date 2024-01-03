# Dockerfile
FROM python:3.8

WORKDIR /app

COPY model.py /app/model.py

RUN pip install mlflow scikit-learn

CMD ["python", "model.py"]
