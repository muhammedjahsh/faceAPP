

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# import streamlit as st
import os
from PIL import Image


import google.generativeai as genai


os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load OpenAI model and get respones

def get_gemini_response1(input, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    prompt = f"From the given image and prompt, extract only the person's name. {input}"
    if input != "":
        response = model.generate_content([prompt, image])
    else:
        response = model.generate_content(["Extract only the person's name from the image.", image])
    return response.text.strip()





    
    