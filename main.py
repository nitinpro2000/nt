import asyncio
import uuid
import os
from dotenv import load_dotenv
from src.agents.control_agent import ControlAgent

def check_environment():
    """Check if required environment variables are set."""
    load_dotenv()
    
    required_vars = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file:")
        print("GOOGLE_API_KEY=your_google_api_key_here")
        print("SERPAPI_API_KEY=your_serpapi_key_here")
        return False
    return True

async def main():
    # Check environment variables
    if not check_environment():
        return
    
    try:
        # Initialize the ControlAgent
        control_agent = ControlAgent()
        
        # Example company and focus points
        company_name = "Tesla"
        focus_points = [
            "Electric vehicle production",
            "Battery technology",
            "Autonomous driving"
        ]
        
        # Compose industry news using the control agent
        print("Composing industry news...")
        results = await control_agent.compose_industry_news(
            company_name=company_name,
            focus_points=focus_points,
            max_results=3,
            time_period="last_month"
        )
        
        # Check if there was an error
        if "error" in results:
            print(f"Error occurred: {results['error']}")
            return
        
        # Print results in a formatted way
        print("\nIndustry News Analysis Results:")
        print("=" * 50)
        print(f"Company: {results['company']}")
        print(f"Industry: {results['industry']}")
        print(f"Session ID: {results['session_id']}")
        print(f"Timestamp: {results['timestamp']}")
        print("=" * 50)
        
        print("\nNews Summary by Focus Point:")
        for focus_summary in results["news_summary"]:
            print(f"\nFocus Point: {focus_summary['focus_point']}")
            print("-" * 30)
            
            for article in focus_summary["articles"]:
                print(f"\nTitle: {article['title']}")
                print(f"URL: {article['url']}")
                print(f"Relevance Score: {article['relevance_score']:.2f}")
                print(f"Snippet: {article['snippet']}")
                print(f"Text Preview: {article['text_snippet']}")
                print("-" * 30)
        
        # Save session history
        control_agent.save_session_history()
        print("\nSession history saved to session_history.json")
        
        # Example of retrieving session history
        session_id = results['session_id']
        session_info = control_agent.get_session_history(session_id)
        if session_info:
            print("\nSession History:")
            print(f"Timestamp: {session_info['timestamp']}")
            print(f"Company: {session_info['company']}")
            print(f"Industry: {session_info['industry']}")
            print(f"Focus Points: {', '.join(session_info['focus_points'])}")
            
    except ImportError as e:
        print(f"Error: Missing required module - {str(e)}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nPlease check:")
        print("1. All required API keys are set in .env file")
        print("2. All dependencies are installed")
        print("3. Internet connection is available")
        print("4. Required directories exist and are writable")

if __name__ == "__main__":
    asyncio.run(main()) 