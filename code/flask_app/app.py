from flask import Flask, render_template, request, jsonify, redirect
import os
import pandas as pd
import numpy as np
import pickle
import os, urllib, cv2, re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image, ImageTk
from itertools import count
import string
import matplotlib.pyplot as plt
from moviepy.editor import *
import pygame
import time
import PyPDF2
import pytesseract as tess
from PIL import Image
from gtts import gTTS
import shutil
tess.pytesseract.tesseract_cmd = r'C:\Users\VAMC\AppData\Local\Programs\Tesseract-OCR\tesseract'

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
    open('output.mp3', 'a').close()

def move_webm_to_static():
    current_dir = './output.webm'
    static_dir = './static/output.webm'
    shutil.move(current_dir, static_dir)

def gif_to_webm(gif_path, webm_path):
    gif_clip = VideoFileClip(gif_path)
    gif_clip.write_videofile(webm_path, codec="libvpx")

def func(data):
    isl_gif=['any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
                    'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
                    'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
                    'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
                    'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
                     'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
                    'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
                    'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
                    'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
                    'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
                    'what are you doing', 'what is the problem', 'what is todays date', 'what is your father do', 'what is your job',
                    'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
                    'where is the bathroom', 'where is the police station', 'you are wrong','address','agra','ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
    'bihar','bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
    'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
    'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
    'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
    'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
    'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
    'voice', 'wednesday', 'weight','please wait for sometime','what is your mobile number','what are you doing','are you busy']

    arr=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r', 's','t','u','v','w','x','y','z']        
    MyText = data
    a = MyText.lower()
    print(MyText)
    for c in string.punctuation:
        a= a.replace(c,"")
    
    if(a.lower() in isl_gif):
        print('true')
        ImageAddress = 'D:/Automatic-Indian-Sign-Language-Translator-ISL-master/ISL_Gifs/'+a+'.gif'
        if os.path.exists(ImageAddress):
            gif_to_webm(ImageAddress, 'output.webm')
                    
    else:
        output_video_path = 'output.webm'
        frame_size = (640, 480)
        frame_rate = 30.0
        frame_duration = 0.8 
        def write_frame_with_duration(frame, duration):
            num_frames = int(frame_rate * duration)
            for _ in range(num_frames):
                out.write(frame)
        try:
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(output_video_path, fourcc,frame_rate,frame_size)
        except:
            print('okk')
        for i in range(len(a)):
            if(a[i] in arr):
                ImageAddress = 'D:/Automatic-Indian-Sign-Language-Translator-ISL-master/letters/'+a[i]+'.jpg'
                img = cv2.imread(ImageAddress)
                if img is not None:
                    # Resize image if needed
                    img = cv2.resize(img, frame_size)
                    try:

                        write_frame_with_duration(img, frame_duration)
                    except:
                        print('okk')

            else:
                continue
        out.release()
        plt.close()


def make_prediction(selected_img_path, selected_que):
    
    def decontractions(phrase):
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"won\’t", "will not", phrase)
        phrase = re.sub(r"can\’t", "can not", phrase)
        phrase = re.sub(r"he\'s", "he is", phrase)
        phrase = re.sub(r"she\'s", "she is", phrase)
        phrase = re.sub(r"it\'s", "it is", phrase)
        phrase = re.sub(r"he\’s", "he is", phrase)
        phrase = re.sub(r"she\’s", "she is", phrase)
        phrase = re.sub(r"it\’s", "it is", phrase)
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"n\’t", " not", phrase)
        phrase = re.sub(r"\’re", " are", phrase)
        phrase = re.sub(r"\’d", " would", phrase)
        phrase = re.sub(r"\’ll", " will", phrase)
        phrase = re.sub(r"\’t", " not", phrase)
        phrase = re.sub(r"\’ve", " have", phrase)
        phrase = re.sub(r"\’m", " am", phrase)
        return phrase

    def text_preprocess(text):
        text = text.lower()
        text = decontractions(text) 
        text = re.sub('[-,:]', ' ', text)
        text = re.sub("(?!<=\d)(\.)(?!\d)", '', text) 
        text = re.sub('[^A-Za-z0-9. ]+', '', text) 
        text = re.sub(' +', ' ', text)
        return text

    
    def load_data_model():
        data = pd.read_csv('./static/mscoco_train2014_k1000_50k.csv')
        tokenizer_50k = pickle.load(open('./static/tokenizer_50k.pkl', 'rb'))
        labelencoder = pickle.load(open('./static/labelencoder.pkl', 'rb'))
        model = tf.keras.models.load_model('./static/model_2lstm_vgg19_50k_1011_50.h5')
        return data, tokenizer_50k, labelencoder, model
    
    def final_function_1(X): 
        
        if (X[0] is None) or ((X[1] is None)):
            return " "
    
        que_clean_text = text_preprocess(X[0])
        que_vector = pad_sequences(tokenizer.texts_to_sequences([que_clean_text]), maxlen=22, padding='post') 
        
        if type(X[1]) == str:
            img = cv2.imread(X[1])
            img = cv2.resize(img,(224,224),interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(X[1])
            img = img.resize((224,224))
            img = np.array(img)

        img_vector = (img/255.0).astype(np.float32)
        predicted_Y = model.predict([que_vector,np.array([img_vector])],verbose=0)
        predicted_class = tf.argmax(predicted_Y, axis=1, output_type=tf.int32)
        predicted_ans = labelencoder.inverse_transform(predicted_class)

        return predicted_ans[0]
    
    data, tokenizer, labelencoder, model = load_data_model()    
    predicted_ans = final_function_1([selected_que, selected_img_path])
    print('**Predicted Answer:**',predicted_ans) 
    return predicted_ans
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/features.html")
def index1():
    return render_template('features.html')

@app.route("/upload.html", methods=['GET','POST'])
def index5():
    
    return render_template('upload.html')



@app.route("/Sign_translation.html", methods=['GET','POST'])
def index2():
    vid='./static/output.webm'
    if os.path.exists(vid):
        try:
            os.remove(vid)
            print(f"{vid} deleted successfully.")
        except OSError as error:
            print("Error deleting image:", error)

    if request.method=='POST':
        vid='./static/output.webm'
        if os.path.exists(vid):
            try:
                os.remove(vid)
                print(f"{vid} deleted successfully.")
            except OSError as error:
                print("Error deleting image:", error)
        text = request.form['text']
        print(text)
        func(text)
        move_webm_to_static()

        print('completed')
    return render_template('Sign_translation.html')

@app.route("/OCR.html", methods=['GET','POST'])
def index3():
    aud='./static/output.mp3'
    if os.path.exists(aud):
        try:
            os.remove(aud)
            print(f"{aud} deleted successfully.")
        except OSError as error:
            print("Error deleting image:", error)
    texter=''
    if request.method=='POST':
        text=request.form['text']
        filee=request.form['file']
        print(type(text))
        print(type(filee))
        

        if len(text)==0:
            image_path = f'F:/major project/input/{filee}'
            if filee.endswith('.pdf'):
                pdf_text = extract_text_from_pdf(image_path)
                texter+=pdf_text
                save_text_to_file(pdf_text, "./static/output.txt")
                tts = gTTS(text=pdf_text, lang='en')
                tts.save("./static/output.mp3")
                create_audio_file()
            elif filee.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_text = extract_text_from_image(image_path)
                texter+=image_text
                save_text_to_file(image_text, "./static/output.txt")
                tts = gTTS(text=image_text, lang='en')
                tts.save("./static/output.mp3")
                create_audio_file()

        if len(filee)==0:
            texter=text
            tts = gTTS(text=texter, lang='en')
            tts.save("./static/output.mp3")
        if len(filee)!=0 and len(text)!=0:
            texter=text
            tts = gTTS(text=texter, lang='en')
            tts.save("./static/output.mp3")

    return render_template('OCR.html', text=texter)
    return render_template('OCR.html')

@app.route("/Clear")
def clear():
    image_path = "./static/input_vqa.png"
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            print("Image deleted successfully.")
        except OSError as error:
            print("Error deleting image:", error)
    return redirect('upload.html')

@app.route("/VQA.html", methods=['GET','POST'])
def index4():
    img='./static/input_vqa.png'
    if os.path.exists(img):
        try:
            os.remove(img)
            print(f"{img} deleted successfully.")
        except OSError as error:
            print("Error deleting image:", error)
    if request.method=='POST':
        name=request.form['file']
        print('heroo', name)
        image_path = f'F:/major project/input/VQA/{name}'
        try:
            img = Image.open(image_path)
            img.save('./static/input_vqa.png')
            print("Image retrieved and saved successfully!")

        except FileNotFoundError:
            print("Error: Image file not found at", image_path)
    image_path='static/input_vqa.png'
    return render_template('VQA.html')

@app.route("/speech_sign.html", methods=['GET','POST'])
def index10():
    vid='./static/output.webm'
    if os.path.exists(vid):
        try:
            os.remove(vid)
            print(f"{vid} deleted successfully.")
        except OSError as error:
            print("Error deleting image:", error)
    if request.method=='POST':
        text=request.form['signtext']
        print(text)
        func(text)
        move_webm_to_static()
        print('completed')
    return render_template('speech_sign.html')
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return make_prediction('./static/input_vqa.png', input)
if __name__ == '__main__':
    app.run(debug=True)
