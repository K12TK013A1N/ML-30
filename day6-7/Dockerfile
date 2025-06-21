# Dockerfile
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY app.py model.pkl /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 5000 and run the app
EXPOSE 5000
CMD ["python", "app.py"]