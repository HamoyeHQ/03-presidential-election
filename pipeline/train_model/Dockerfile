FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && \
     apt-get -y --no-install-recommends install \
     libgomp1

RUN pip install -U scikit-learn numpy pandas scipy

COPY train_model.py ./train_model.py

ENTRYPOINT [ "python", "train_model.py" ]
