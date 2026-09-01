FROM python:3.11
WORKDIR /code
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt
COPY . .
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
