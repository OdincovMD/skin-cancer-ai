FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY /app /app
COPY requirements.txt requirements.txt
EXPOSE 9000

RUN pip3 install -r requirements.txt

CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "9000", "--reload", "--workers", "2"]