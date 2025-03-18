# CV Analyzer API

A FastAPI service that extracts and analyzes information from CVs/resumes using OpenAI's API.

## Features

- PDF text extraction with PyPDF2
- OpenAI-powered CV analysis and structured data extraction
- Caching system for efficient processing
- Docker support for easy deployment
- Comprehensive logging

## API Endpoints

- `POST /upload-cv/`: Upload a PDF resume for analysis
- `GET /`: Health check endpoint

## Requirements

- Python 3.12+
- FastAPI
- OpenAI Python client
- PyPDF2
- Docker (optional)

## Environment Variables

Configure the following variables in a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1/  # Optional, defaults to OpenAI's endpoint
```

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

## Docker Deployment

```bash
# Build the image
docker build -t cv-analyzer .

# Run the container
docker run -p 8000:8000 --env-file .env cv-analyzer
```

## Using the API

Upload a PDF CV:

```bash
curl -X POST -F "file=@path/to/your/cv.pdf" http://localhost:8000/upload-cv/
```

## Cache System

The application caches results for 7 days to improve performance for repeated requests with the same CV file.

## Docker Hub

This image is available on Docker Hub:

```bash
docker pull username/cv-analyzer:latest
```

Run with:

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here username/cv-analyzer:latest
```