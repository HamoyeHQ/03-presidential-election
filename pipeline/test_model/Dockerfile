FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy

COPY test_model.py ./test_model.py

ENTRYPOINT [ "python", "test_model.py" ]
