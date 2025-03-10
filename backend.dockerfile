FROM python:3.12-bookworm

WORKDIR /app

# copy the requirements file
COPY requirements.backend.txt /app/
COPY main.py /app/
COPY src /app/src
COPY checkpoints /app/checkpoints


# install dependencies
RUN pip install --upgrade pip \
&& pip install --no-cache-dir -r requirements.backend.txt

RUN ls -la /app >&2

# command to run on container start
ENTRYPOINT [ "fastapi", "run", "./main.py", "--port", "8000" ]