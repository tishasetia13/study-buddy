# Start from a lightweight official Python image
FROM python:3.14-slim

# All following commands run from /app inside the container
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your project code + data files
COPY . .

# Document that this container listens on port 8000
EXPOSE 8000

# The command that runs when the container starts
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}