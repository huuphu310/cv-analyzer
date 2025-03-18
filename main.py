from fastapi import FastAPI, File, UploadFile, HTTPException
import openai
import tempfile
import os
import PyPDF2
import logging
import json
from typing import Dict, Any
import hashlib
from pathlib import Path
import pickle
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cv_processor.log")
    ]
)
logger = logging.getLogger("cv_processor")

# Create cache directory if it doesn't exist
CACHE_DIR = Path("cv_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY = 60 * 60 * 24 * 7  # 7 days in seconds


app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

client = openai.Client(api_key=OPENAI_API_KEY, base_url=BASE_URL)

def get_file_hash(content: bytes) -> str:
    """Generate a unique hash for file content"""
    return hashlib.md5(content).hexdigest()


def get_from_cache(file_hash: str) -> Dict[Any, Any]:
    """Get cached result by file hash if it exists and is not expired"""
    cache_file = CACHE_DIR / f"{file_hash}.pkl"

    if not cache_file.exists():
        logger.debug(f"Cache miss: No cache file for hash {file_hash}")
        return None

    try:
        with open(cache_file, "rb") as f:
            timestamp, result = pickle.load(f)

        # Check if cache is expired
        if time.time() - timestamp > CACHE_EXPIRY:
            logger.info(f"Cache expired for hash {file_hash}")
            cache_file.unlink()
            return None

        logger.info(f"Cache hit: Using cached result for hash {file_hash}")
        return result

    except Exception as e:
        logger.error(f"Error reading cache: {str(e)}", exc_info=True)
        return None


def save_to_cache(file_hash: str, result: Dict[Any, Any]) -> None:
    """Save result to cache with current timestamp"""
    cache_file = CACHE_DIR / f"{file_hash}.pkl"

    try:
        with open(cache_file, "wb") as f:
            pickle.dump((time.time(), result), f)
        logger.info(f"Saved result to cache for hash {file_hash}")
    except Exception as e:
        logger.error(f"Error saving to cache: {str(e)}", exc_info=True)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from PDF file"""
    text = ""
    try:
        logger.info(f"Starting text extraction from PDF: {file_path}")
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            page_count = len(pdf_reader.pages)
            logger.info(f"PDF has {page_count} pages")

            for i, page in enumerate(pdf_reader.pages):
                logger.debug(f"Extracting text from page {i + 1}/{page_count}")
                page_text = page.extract_text()
                text += page_text + "\n"

        logger.info(f"Text extraction complete. Extracted {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def generate_json_from_text(cv_text: str) -> Dict[Any, Any]:
    """Send CV text to OpenAI and get JSON response"""
    try:
        logger.info(f"Sending CV text to OpenAI API (length: {len(cv_text)} chars)")

        response = client.chat.completions.create(
            model="auto",
            messages=[
                {"role": "system",
                 "content": "You are an expert CV analyzer. Extract structured information from the given CV and return it in JSON format."},
                {"role": "user",
                 "content": f"Analyze this CV text and extract structured information as JSON:\n\n{cv_text}"}
            ],
            response_format={"type": "json_object"}
        )

        logger.info("Received response from OpenAI API")

        # Parse the JSON string into a Python dictionary
        content = response.choices[0].message.content
        json_data = json.loads(content)
        logger.info(f"Successfully parsed JSON response with {len(json_data)} top-level fields")

        return json_data
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
        raise Exception(f"OpenAI API error: {str(e)}")


@app.post("/upload-cv/")
async def upload_cv(file: UploadFile = File(...)):
    """Receive CV file, validate PDF, extract text, and analyze with OpenAI"""
    logger.info(f"Received file upload: {file.filename}")

    # Validate file is PDF
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Rejected non-PDF file: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        # Read file content
        content = await file.read()

        # Generate hash for file content
        file_hash = get_file_hash(content)
        logger.info(f"Generated hash for file: {file_hash}")

        # Check cache for existing result
        cached_result = get_from_cache(file_hash)
        if cached_result:
            logger.info(f"Returning cached result for file {file.filename}")
            return cached_result

        # Save file temporarily
        logger.info("Saving uploaded file to temporary location")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"Temporary file saved at: {temp_file_path}")

        # Extract text from PDF
        cv_text = extract_text_from_pdf(temp_file_path)

        # Clean up temporary file
        logger.info(f"Removing temporary file: {temp_file_path}")
        os.unlink(temp_file_path)

        # If text extraction successful, generate JSON
        if cv_text:
            logger.info("Starting JSON generation from extracted text")
            extracted_json = generate_json_from_text(cv_text)
            logger.info("Successfully generated JSON from CV")

            # Save result to cache
            save_to_cache(file_hash, extracted_json)

            return extracted_json
        else:
            logger.error("Failed to extract text from PDF")
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF")

    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}", exc_info=True)
        return {"error": str(e)}