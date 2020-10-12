FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas imbalanced-learn

COPY preprocess_data.py ./preprocess_data.py
COPY president-1976-2016.csv ./president-1976-2016.csv

ENTRYPOINT [ "python", "preprocess_data.py" ]
