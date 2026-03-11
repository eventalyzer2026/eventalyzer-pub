FROM python:3.12-slim

WORKDIR /eventalyzer

COPY . ./app/src

RUN pip install -r ./app/src/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.src.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
