from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import json
from pathlib import Path

from src.plugins.news_agent_plugin import NewsAgentPlugin

class ControlAgent:
    def __init__(self):
        """Initialize the ControlAgent with available plugins."""
        self.plugins = {
            "news_agent": NewsAgentPlugin()
        }
        self.session_history = {}
        
    def register_plugin(self, name: str, plugin: Any):
        """Register a new plugin with the control agent."""
        self.plugins[name] = plugin
        
    async def compose_industry_news(
        self,
        company_name: str,
        focus_points: List[str],
        session_id: Optional[str] = None,
        max_results: int = 5,
        time_period: str = "last_month"
    ) -> Dict[str, Any]:
        """
        Compose industry news by orchestrating multiple plugin operations.
        
        Args:
            company_name: Name of the company to analyze
            focus_points: List of focus points for news analysis
            session_id: Optional session ID for tracking
            max_results: Maximum number of results to return
            time_period: Time period for news (e.g., "last_week", "last_month")
            
        Returns:
            Dict containing composed news analysis
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        try:
            # Step 1: Extract keywords using NewsAgentPlugin
            keywords_result = await self.plugins["news_agent"].extract_search_keywords(
                company_name,
                focus_points
            )
            
            # Step 2: Fetch news articles
            articles = self.plugins["news_agent"].fetch_industry_news(
                keywords_result["industry"],
                keywords_result["keywords"]
            )
            
            # Step 3: Process and store articles
            processed_df = await self.plugins["news_agent"].process_articles(
                articles,
                session_id
            )
            
            # Store in vector database
            self.plugins["news_agent"].store_in_vector_db(processed_df)
            
            # Step 4: Compose final results
            composed_results = {
                "company": company_name,
                "industry": keywords_result["industry"],
                "focus_points": focus_points,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "news_summary": []
            }
            
            # Get relevant news for each focus point
            for focus_point in focus_points:
                query = f"{company_name} {focus_point} {time_period}"
                results = self.plugins["news_agent"].retrieve_from_vector_db(
                    query,
                    n_results=max_results
                )
                
                focus_point_summary = {
                    "focus_point": focus_point,
                    "articles": []
                }
                
                for result in results:
                    article_summary = {
                        "title": result["metadata"]["title"],
                        "url": result["metadata"]["link"],
                        "snippet": result["metadata"]["snippet"],
                        "relevance_score": 1 - result["distance"],
                        "text_snippet": result["text"][:200] + "..."
                    }
                    focus_point_summary["articles"].append(article_summary)
                
                composed_results["news_summary"].append(focus_point_summary)
            
            # Store session history
            self.session_history[session_id] = {
                "timestamp": datetime.now().isoformat(),
                "company": company_name,
                "industry": keywords_result["industry"],
                "focus_points": focus_points
            }
            
            return composed_results
            
        except Exception as e:
            error_response = {
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            return error_response
    
    def get_session_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session history for a given session ID."""
        return self.session_history.get(session_id)
    
    def save_session_history(self, filepath: str = "session_history.json"):
        """Save session history to a JSON file."""
        history_file = Path(filepath)
        with history_file.open("w") as f:
            json.dump(self.session_history, f, indent=2)
    
    def load_session_history(self, filepath: str = "session_history.json"):
        """Load session history from a JSON file."""
        history_file = Path(filepath)
        if history_file.exists():
            with history_file.open("r") as f:
                self.session_history = json.load(f) 