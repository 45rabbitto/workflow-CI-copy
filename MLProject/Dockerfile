FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    cloudpickle==3.1.2 \
    numpy==1.26.2 \
    psutil==7.2.2 \
    scikit-learn==1.5.1 \
    scipy==1.16.3 \
    matplotlib==3.8.2 \
    pandas==2.1.4

COPY model /app/model

EXPOSE 8080

ENTRYPOINT ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080"]
