# test_brain.py
from brain_of_the_doctor import encode_image, analyze_image_with_query

# Test with a sample image
encoded = encode_image("test_image.jpg")
print(f"Image encoded: {encoded[:50]}...")

# Test analysis (will need GROQ_API_KEY set)
result = analyze_image_with_query(
    "What's in this image?",
    encoded,
    model="llama-3-70b-8192"
)
print("Analysis result:", result)