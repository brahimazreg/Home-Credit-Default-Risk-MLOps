FROM python:3.11-slim

# prevent python buffering logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install dependencies first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .


# expose FastAPI port
EXPOSE 8000

# run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]