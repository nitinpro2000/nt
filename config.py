import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# ChromaDB settings
CHROMA_PERSIST_DIRECTORY = "data/chroma_db"
CHROMA_COLLECTION_NAME = "news_articles"

# News processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_ARTICLES_PER_SEARCH = 10

# Model settings
GEMINI_MODEL = "gemini-pro"
EMBEDDING_MODEL = "models/embedding-001"

# SerpAPI settings
SERPAPI_SEARCH_ENGINE = "google"
SERPAPI_NUM_RESULTS = 10 