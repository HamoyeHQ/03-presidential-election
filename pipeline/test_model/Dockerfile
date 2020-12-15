FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && \
     apt-get -y --no-install-recommends install \
     libgomp1

RUN pip install -U scikit-learn numpy scipy

COPY test_model.py ./test_model.py

ENTRYPOINT [ "python", "test_model.py" ]
