from typing import List, Dict, Any
import semantic_kernel as sk
from semantic_kernel.connectors.ai.google_palm import GooglePalmTextCompletion
from semantic_kernel.connectors.ai.google_palm import GooglePalmTextEmbedding
from serpapi import GoogleSearch
import chromadb
from chromadb.config import Settings
import pandas as pd
from datetime import datetime
from semantic_kernel.template_engine.prompt_template_engine import PromptTemplateEngine
import json
from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import EmbeddingGeneratorBase

from src.config.config import (
    GOOGLE_API_KEY,
    SERPAPI_API_KEY,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_ARTICLES_PER_SEARCH,
    GEMINI_MODEL,
    EMBEDDING_MODEL
)
from src.utils.text_processing import chunk_text, scrape_article, generate_unique_id

class NewsAgentPlugin:
    def __init__(self):
        # Initialize Semantic Kernel
        self.kernel = sk.Kernel()
        
        # Add Google Gemini model
        self.kernel.add_text_completion_service(
            "gemini",
            GooglePalmTextCompletion(
                api_key=GOOGLE_API_KEY,
                model_id=GEMINI_MODEL
            )
        )
        
        # Add Google embeddings
        self.kernel.add_text_embedding_generation_service(
            "google-embeddings",
            GooglePalmTextEmbedding(
                api_key=GOOGLE_API_KEY,
                model_id=EMBEDDING_MODEL
            )
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=CHROMA_PERSIST_DIRECTORY
        ))
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )

    async def extract_search_keywords(self, company_name: str, focus_points: List[str]) -> Dict[str, Any]:
        """Extract industry and search keywords using LLM."""
        prompt = f"""
        Given the company name '{company_name}' and the following focus points:
        {', '.join(focus_points)}
        
        Please provide:
        1. The industry this company belongs to
        2. A list of relevant keywords for each focus point that would be useful for finding news articles
        
        Format the response as a JSON with the following structure:
        {{
            "industry": "industry name",
            "keywords": {{
                "focus_point_1": ["keyword1", "keyword2", ...],
                "focus_point_2": ["keyword1", "keyword2", ...],
                ...
            }}
        }}
        """
        prompt_template_config = sk.PromptTemplateConfig()
        prompt_template = sk.PromptTemplate(
            prompt,
            template_engine=PromptTemplateEngine(),
            prompt_config=prompt_template_config
        )
        semantic_func = self.kernel.register_semantic_function(
            "news_agent",
            "extract_keywords",
            sk.SemanticFunctionConfig(prompt_template_config, prompt_template)
        )
        response = await self.kernel.run_async(semantic_func)
        print('LLM output:', response.result)
        try:
            if not response.result or not response.result.strip():
                raise ValueError('Empty response from LLM')
            return json.loads(response.result)
        except Exception as e:
            print('Error parsing LLM output:', e)
            return {"industry": None, "keywords": {}}

    def fetch_industry_news(self, industry: str, keywords: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Fetch news articles using SerpAPI."""
        articles = []
        
        for focus_point, keyword_list in keywords.items():
            search_query = f"{industry} {' '.join(keyword_list)} news"
            
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": SERPAPI_API_KEY,
                "num": MAX_ARTICLES_PER_SEARCH
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "organic_results" in results:
                for result in results["organic_results"]:
                    articles.append({
                        "url": result.get("link", ""),
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "keywords": keyword_list,
                        "focus_point": focus_point
                    })
        
        return articles

    async def process_articles(self, articles: List[Dict[str, str]], session_id: str) -> pd.DataFrame:
        """Process articles and prepare for vector storage."""
        processed_data = []
        
        for article in articles:
            # Scrape article content
            content = scrape_article(article["url"])
            if not content:
                continue
            
            # Chunk the content
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            
            # Generate article ID
            article_id = generate_unique_id()
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = generate_unique_id()
                
                # Generate embedding for the chunk
                embedding_service = self.kernel.get_ai_service(EmbeddingGeneratorBase, 'google-embeddings')
                embedding = await embedding_service.generate_embeddings_async(chunk)
                
                processed_data.append({
                    "link": article["url"],
                    "title": article["title"],
                    "snippet": article["snippet"],
                    "keywords": article["keywords"],
                    "session_id": session_id,
                    "category": article["focus_point"],
                    "chunked_text": chunk,
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "embedding": embedding.result
                })
        
        return pd.DataFrame(processed_data)

    def store_in_vector_db(self, df: pd.DataFrame):
        """Store processed articles in ChromaDB."""
        for _, row in df.iterrows():
            self.collection.add(
                ids=[row["chunk_id"]],
                embeddings=[row["embedding"]],
                metadatas=[{
                    "link": row["link"],
                    "title": row["title"],
                    "snippet": row["snippet"],
                    "keywords": row["keywords"],
                    "session_id": row["session_id"],
                    "category": row["category"],
                    "article_id": row["article_id"]
                }],
                documents=[row["chunked_text"]]
            )

    def retrieve_from_vector_db(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from ChromaDB."""
        # Generate query embedding
        embedding_service = self.kernel.get_ai_service(EmbeddingGeneratorBase, 'google-embeddings')
        query_embedding = embedding_service.generate_embeddings(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return [{
            "text": doc,
            "metadata": meta,
            "distance": dist
        } for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )] 