import re
from typing import List
import uuid
from bs4 import BeautifulSoup
import requests

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        start = end - overlap

    return chunks

def scrape_article(url: str) -> str:
    """Scrape article content from URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean and normalize text
        text = clean_text(text)
        
        return text
    except Exception as e:
        print(f"Error scraping article {url}: {str(e)}")
        return ""

def generate_unique_id() -> str:
    """Generate a unique ID for articles and chunks."""
    return str(uuid.uuid4()) 