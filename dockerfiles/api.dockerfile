# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /src/group_99/api

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY src/group_99/main.py .
COPY src/group_99/api/frontend.py .
COPY src/group_99/api/model.py .
COPY models/best-model-epoch=04-val_loss=0.77.ckpt .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Start both the FastAPI backend and Streamlit frontend
CMD uvicorn app.backend:app --host 0.0.0.0 --port 8000 & \
    streamlit run app/frontend.py --server.port 8501 --server.address 0.0.0.0
