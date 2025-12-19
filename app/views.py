from flask import render_template,request
import os
import cv2
from app.faceRecognition import faceRecognitionPipeline
import matplotlib.image as matimg
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# import streamlit as st
import os
from PIL import Image


import google.generativeai as genai

UPLOAD_FOLDER  = "static/upload"




def index():
    return render_template('index.html')

def app():
    return render_template("app.html")

def gender():
    if request.method == "POST":
        f = request.files['image_name']
        filename = f.filename
        # save our image into upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)   # this save the files into upload folder
         # get predictions
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)

        # generate report
        report = []
        for i , obj in enumerate(predictions):
            gray_image = obj['roi'] # grayscale image (array)
            eigen_image = obj['eig_img'].reshape(100,100) # eigen image (array)
            gender_name = obj['prediction name'] # name 
            score = round(obj['score']*100,2) # probability score
            
            # save grayscale and eigne in predict folder
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image,cmap='gray')
            matimg.imsave(f'./static/predict/{eig_image_name}',eigen_image,cmap='gray')
            
            # save report 
            report.append([gray_image_name,
                           eig_image_name,
                           gender_name,
                           score])
        return render_template("genderApp.html",fileupload = True,report=report)  # post request
            

    return render_template("genderApp.html",fileupload = False)  # get request


# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ## Function to load OpenAI model and get respones

# from flask import request
# import os
# from PIL import Image
# import google.generativeai as genai  # Make sure this is correctly configured
# # from your_module import extract_name_from_image  # You need to define this

# UPLOAD_FOLDER = 'static/uploads'

# extracted_names = []
# @app.route('/', methods=['GET', 'POST'])
# def get_name():
#     if request.method == "POST":
#         f = request.files['image_name']
#         filename = f.filename
        
#         # Save image to upload folder
#         path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         f.save(path)

#         image = Image.open(path)
#         input_prompt = request.form.get("user_prompt", "Extract the name from this document")

#         # Extract name
#         model = genai.GenerativeModel('gemini-pro-vision')
#         response = model.generate_content([input_prompt, image])
#         name = response.text.strip()

#         # Add name to list
#         extracted_names.append(name)
#     return render_template("genderApp.html",fileupload = True,name=extracted_names)

