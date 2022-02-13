FROM python:3.9.10-slim
LABEL maintainer="sergeyz06@yandex.ru"
COPY model.dill /app/model.dill
COPY run_server.py /app/run_server.py
COPY requirements.txt /app/requirements.txt
WORKDIR /app
EXPOSE 8180
RUN apt update
RUN apt install -y python3-virtualenv
RUN virtualenv /app/venv
RUN /app/venv/bin/pip3 install -r /app/requirements.txt
ENTRYPOINT /app/venv/bin/python3 run_server.py
