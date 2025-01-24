# Use an official Python runtime as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY ../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY ../ .

# Expose port 80 (optional, adjust as needed)
EXPOSE 80

# Command to run the application
CMD ["python", "app.py"]
