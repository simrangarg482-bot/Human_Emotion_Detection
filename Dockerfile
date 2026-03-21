# Use Python 3.10
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements.txt first
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Now copy the rest of the code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]