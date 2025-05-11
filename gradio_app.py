# gradio_app.py
import os
import time
import gradio as gr
from dotenv import load_dotenv
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs
from brain_of_the_doctor import analyze_image_with_query
from groq import Groq

# Load environment variables
load_dotenv()

# ========== RED THEME CONFIGURATION ==========
red_theme = gr.themes.Default(
    primary_hue="red",
    secondary_hue="red",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "Arial", "sans-serif"],
).set(
    button_primary_background_fill="#d32f2f",
    button_primary_background_fill_dark="#b71c1c",
    button_primary_text_color="#ffffff",
    button_primary_background_fill_hover="#f44336",
    slider_color="#d32f2f",
    block_title_text_color="#d32f2f",
    block_label_text_color="#d32f2f",
)

# Language and voice configuration
LANGUAGE_CONFIG = {
    'en': {'voice': 'Rachel', 'model': 'eleven_turbo_v2', 'prompt': "Provide medical analysis in English"},
    'hi': {'voice': 'Aria', 'model': 'eleven_multilingual_v2', 'prompt': "हिंदी में चिकित्सा विश्लेषण प्रदान करें"},
    'es': {'voice': 'Isabella', 'model': 'eleven_multilingual_v2', 'prompt': "Proporcione análisis médico en español"},
    'fr': {'voice': 'Claude', 'model': 'eleven_multilingual_v2', 'prompt': "Fournir une analyse médicale en français"},
    'default': {'voice': 'Rachel', 'model': 'eleven_turbo_v2', 'prompt': "Provide medical analysis"}
}

def detect_language(audio_path):
    """Detect language from audio using Whisper"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language=None  # Auto-detect
            )
        return result.language[:2]  # Return 2-letter code
    except Exception as e:
        print(f"Language detection error: {e}")
        return 'en'  # Default to English

def process_inputs(audio_filepath, image_filepath):
    try:
        # 1. Detect language
        lang_code = detect_language(audio_filepath) if audio_filepath else 'en'
        lang_config = LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG['default'])
        
        # 2. Transcribe audio
        patient_query = transcribe_with_groq(audio_filepath) if audio_filepath else "No audio provided"
        
        # 3. Medical analysis
        diagnosis = analyze_image_with_query(
            query=f"{lang_config['prompt']}\nPatient says: {patient_query}",
            image_path=image_filepath
        )
        
        # 4. Generate voice response
        voice_path = f"diagnosis_{int(time.time())}.mp3"
        text_to_speech_with_elevenlabs(
            input_text=diagnosis,
            output_filepath=voice_path,
            voice=lang_config['voice'],
            model=lang_config['model'],
            language=lang_code
        )
        
        return patient_query, diagnosis, voice_path
        
    except Exception as e:
        error_msg = f"System error: {str(e)}"
        return error_msg, error_msg, None

# ========== INTERFACE ==========
with gr.Blocks(
    title="AI Doctor System",
    theme=red_theme,
    css="""
        .diagnosis {color: #b71c1c; font-weight: bold;}
        #submit-btn {background: linear-gradient(to right, #d32f2f, #b71c1c);}
        .red-border {border: 1px solid #d32f2f !important;}
    """
) as app:
    
    gr.Markdown("""
    <div style='text-align: center; border-bottom: 2px solid #d32f2f; padding-bottom: 10px;'>
    <h1 style='color: #d32f2f;'>🩺 AI Multilingual Doctor</h1>
    <p>Describe symptoms in your language and get diagnosis in same language</p>
    </div>
    """)
    
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Record Symptoms",
            elem_classes="red-border"
        )
        image_input = gr.Image(
            type="filepath",
            label="Upload Medical Image",
            elem_classes="red-border"
        )
    
    submit_btn = gr.Button(
        "Analyze",
        variant="primary",
        elem_id="submit-btn"
    )
    
    with gr.Column():
        transcription = gr.Textbox(
            label="Patient's Description",
            elem_classes="red-border"
        )
        diagnosis = gr.Textbox(
            label="Doctor's Diagnosis",
            elem_classes="diagnosis"
        )
        voice_output = gr.Audio(
            label="Voice Diagnosis",
            autoplay=True
        )

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[transcription, diagnosis, voice_output]
    )

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )