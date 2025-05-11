# brain_of_the_doctor.py
import os
import logging
from groq import Groq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def analyze_image_with_query(query: str, image_path: str = None) -> str:
    """
    Analyze medical query with GROQ's LLM
    
    Args:
        query: The medical query to analyze
        image_path: Optional path to medical image
    
    Returns:
        str: Medical analysis response
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Prepare messages
        messages = [{
            "role": "user",
            "content": query
        }]
        
        # Get response
        response = client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Medical analysis error: {e}")
        return "Medical analysis unavailable currently. Please describe your symptoms in detail."

# Test function
if __name__ == "__main__":
    os.environ["GROQ_API_KEY"] = "your-api-key-here"
    test_response = analyze_image_with_query(
        "I have a rash on my arm. What could this be?"
    )
    print("Test analysis:", test_response)