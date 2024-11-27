# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Fix some problems with the docker image
RUN apt-get update && apt-get install -y libgl1-mesa-dev ffmpeg libsm6 libxext6 wget

# Fix some problems with the model path
RUN mkdir -p /root/.cache/kagglehub/models/gbaonr/best5/pyTorch/default/1/ && \
    wget -O /root/.cache/kagglehub/models/gbaonr/best5/pyTorch/default/1/best5.pt \
    https://www.kaggle.com/api/v1/models/gbaonr/best5/pyTorch/default/1/download/best5.pt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
