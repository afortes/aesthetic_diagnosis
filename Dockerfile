FROM python:3.12-slim

WORKDIR /app

COPY requirements-search.txt .
RUN pip install --no-cache-dir -r requirements-search.txt

COPY search.py .

EXPOSE 8080

CMD ["streamlit", "run", "search.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
