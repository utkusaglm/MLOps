# backend/Dockerfile
FROM --platform=linux/x86_64 python:3.9
RUN python -m pip install --upgrade pip

RUN pip install tensorflow==2.6.2

WORKDIR /app

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "object-detection:app", "--host", "0.0.0.0", "--port", "8000"]