from gtts import gTTS
import os
import pygame
import time
import PyPDF2
import pytesseract as tess
from PIL import Image


# Set the path to the Tesseract executable
tess.pytesseract.tesseract_cmd = r'C:\Users\Chandhana Reddy\AppData\Local\Programs\Tesseract-OCR\tesseract'

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = tess.image_to_string(img)
    return text

def save_text_to_file(text, filename):
    with open(filename, "w") as file:
        file.write(text)

def create_audio_file():
    # Your code to create the audio file goes here
    # For demonstration purposes, we'll create an empty file named 'audio.wav'
    open('output.mp3', 'a').close()

def play_sound(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Assuming 'inputfile' is a function that gets the file name from the user
file = "download.png"
filename = "output.txt"
filenames = 'output.mp3'


if file.endswith('.pdf'):
    pdf_text = extract_text_from_pdf(file)
    save_text_to_file(pdf_text, filename)
    tts = gTTS(text=pdf_text, lang='en')
    tts.save("output.mp3")
    create_audio_file()
    play_sound(filenames)

elif file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
    image_text = extract_text_from_image(file)
    save_text_to_file(image_text, filename)
    tts = gTTS(text=image_text, lang='en')
    tts.save("output.mp3")
    create_audio_file()
    play_sound(filenames)

else:
    print("Unsupported file format. Please provide a PDF or an image file.")
