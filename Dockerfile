FROM python:3.11-slim

WORKDIR /app

COPY backend ./backend
COPY frontend ./frontend

RUN pip install --no-cache-dir -r backend/requirements.txt

ENV PORT=8080
EXPOSE 8080

CMD ["python", "backend/app.py"]
