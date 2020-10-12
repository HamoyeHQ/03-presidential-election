FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy

COPY train_model.py ./train_model.py

ENTRYPOINT [ "python", "train_model.py" ]
