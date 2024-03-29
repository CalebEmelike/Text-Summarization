FROM python:3.8-slim-buster

# Set the working directory in the container
RUN apt-get update -y && apt install awscli -y
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

CMD ["python", "app.py"]