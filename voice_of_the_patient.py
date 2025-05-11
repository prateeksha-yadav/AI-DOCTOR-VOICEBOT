# voice_of_the_patient.py
import os
import logging
from groq.client import Groq  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def transcribe_with_groq(audio_filepath: str, language: str = None) -> str:
    """
    Convert speech to text using GROQ's Whisper
    
    Args:
        audio_filepath: Path to audio file
        language: Optional language code (None for auto-detect)
    
    Returns:
        str: Transcribed text
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        with open(audio_filepath, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language=language
            )
        
        return result.text
        
    except Exception as e:
        logging.error(f"Audio transcription error: {e}")
        return "Audio transcription failed. Please try again."

# Test function
if __name__ == "__main__":
    os.environ["GROQ_API_KEY"] = "your-api-key-here"
    test_transcription = transcribe_with_groq("test_audio.mp3")
    print("Test transcription:", test_transcription)