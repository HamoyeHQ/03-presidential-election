FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas 

COPY preprocess_data.py ./preprocess_data.py
COPY new_vehicle.csv ./new_vehicle.csv

ENTRYPOINT [ "python", "preprocess_data.py" ]
