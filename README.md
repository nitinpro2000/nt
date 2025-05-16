# News Agent with Semantic Kernel

This project implements a news agent using Microsoft's Semantic Kernel, Google's Gemini model, and ChromaDB for vector storage.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following environment variables:
```
GOOGLE_API_KEY=your_google_api_key
SERPAPI_API_KEY=your_serpapi_key
```

## Project Structure

- `src/`
  - `plugins/` - Contains the NewsAgentPlugin
  - `config/` - Configuration files
  - `utils/` - Utility functions
  - `main.py` - Main application entry point

## Usage

Run the main application:
```bash
python src/main.py
```

## Features

- Extract search keywords from company information
- Fetch industry news using SerpAPI
- Scrape and process news articles
- Store and retrieve information using ChromaDB
- Semantic search capabilities using Google's Gemini model 