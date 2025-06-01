FROM python:3.10-slim
WORKDIR /app

RUN pip install kafka-python

COPY src/simulator/simulate.py .

CMD ["python3", "simulate.py"]