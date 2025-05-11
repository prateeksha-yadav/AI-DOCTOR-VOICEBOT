# voice_of_the_doctor.py
import os
import logging
from elevenlabs.client import ElevenLabs
from elevenlabs import save as elevenlabs_save
from gtts import gTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def text_to_speech_with_elevenlabs(
    input_text: str,
    output_filepath: str,
    voice: str = "Rachel",
    model: str = "eleven_turbo_v2",
    language: str = "en"
) -> str:
    """
    Generate speech from text using ElevenLabs API with fallback to gTTS
    
    Args:
        input_text: Text to convert to speech
        output_filepath: Path to save the audio file
        voice: Voice to use (default: "Rachel")
        model: ElevenLabs model to use (default: "eleven_turbo_v2")
        language: Language code for fallback TTS (default: "en")
    
    Returns:
        Path to the generated audio file or None if failed
    """
    try:
        # Verify API key
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API key not found in environment variables")
        
        # Initialize client
        client = ElevenLabs(api_key=api_key)
        
        # Generate speech
        audio = client.generate(
            text=input_text,
            voice=voice,
            model=model,
            voice_settings={
                "stability": 0.35,
                "similarity_boost": 0.85
            }
        )
        
        # Save audio file
        elevenlabs_save(audio, output_filepath)
        logging.info(f"Audio successfully generated and saved to {output_filepath}")
        return output_filepath
        
    except Exception as e:
        logging.error(f"ElevenLabs TTS failed: {e}. Attempting gTTS fallback...")
        try:
            # Fallback to gTTS
            tts = gTTS(
                text=input_text,
                lang=language[:2],  # Use first 2 chars of language code
                slow=False
            )
            tts.save(output_filepath)
            logging.info(f"gTTS fallback audio saved to {output_filepath}")
            return output_filepath
        except Exception as tts_error:
            logging.error(f"gTTS fallback also failed: {tts_error}")
            return None

# Example usage (for testing)
if __name__ == "__main__":
    # Test ElevenLabs TTS
    result = text_to_speech_with_elevenlabs(
        input_text="This is a test of the emergency voice system",
        output_filepath="test_output.mp3",
        voice="Rachel",
        model="eleven_turbo_v2"
    )
    print(f"Voice generation result: {result}")