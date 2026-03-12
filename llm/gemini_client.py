"""
Gemini Client - Handles communication with Google Gemini 1.5 Flash API.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


def get_gemini_client():
    """Initialize and return a Gemini 1.5 Flash model instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in your .env file."
        )
    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
    )

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
    )
    return model


def call_gemini(prompt: str, model=None) -> str:
    """
    Send a prompt to Gemini and return the text response.

    Args:
        prompt: The text prompt to send.
        model: Optional pre-initialized model. If None, creates a new one.

    Returns:
        The model's text response.
    """
    if model is None:
        model = get_gemini_client()

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"
