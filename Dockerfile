# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Initialize and update git submodules
RUN git submodule update --init --recursive

# Copy the rest of the application code into the container at /app
COPY . /app

# Expose port 5000 (or the port your application runs on)
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
