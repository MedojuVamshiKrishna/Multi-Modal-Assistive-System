import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
import imageio
import matplotlib.pyplot as plt
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

    r = sr.Recognizer() 
    while(1):
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                a = MyText.lower()
                print(MyText)
                for c in string.punctuation:
                    a= a.replace(c,"")
                if(a.lower()=='goodbye' or a.lower()=='good bye' or a.lower()=='bye'):
                    print("oops!Time To say good bye")
                    break
                elif(a.lower() in isl_gif):
                    print('true')
                    ImageAddress = 'ISL_Gifs/'+a+'.gif'
                    ImageItself = Image.open(ImageAddress)
                    gif = imageio.mimread(ImageAddress)
                    for frame in gif:
                        plt.imshow(frame)
                        plt.draw()
                        plt.pause(0.1)
                    plt.show(block=True)
                    # ImageNumpyFormat = np.asarray(ImageItself)
                    # plt.imshow(ImageNumpyFormat)
                    # plt.draw()
                    # plt.pause(0.8)
                    class ImageLabel(tk.Label):
                        def load(self, im):
                            if isinstance(im, str):
                                im = Image.open(im)
                            self.loc = 0
                            self.frames = []
                            try:
                                for i in count(1):
                                    self.frames.append(ImageTk.PhotoImage(im.copy()))
                                    im.seek(i)
                                    print(im.seek(i))
                                    print(self.frames)
                            except EOFError:
                                print('excepted')
                                pass
                            try:
                                self.delay = im.info['duration']
                            except:
                                self.delay = 100
                            if len(self.frames) == 1:
                                self.config(image=self.frames[0])
                            else:
                                self.next_frame()
                            def unload(self):
                                self.config(image=None)
                                self.frames = None

                            def next_frame(self):
                                if self.frames:
                                    self.loc += 1
                                    self.loc %= len(self.frames)
                                    self.config(image=self.frames[self.loc])
                                    self.after(self.delay, self.next_frame)
                            print(self.frames)
                            root = tk.Tk()
                            lbl = ImageLabel(root)
                            lbl.pack()
                            lbl.load(r'ISL_Gifs/{0}.gif'.format(a.lower()))
                            root.mainloop()
                else:
                    output_video_path = 'output_video.webm'
                    frame_size = (640, 480)
                    frame_rate = 30.0
                    frame_duration = 0.8 
                    def write_frame_with_duration(frame, duration):
                        num_frames = int(frame_rate * duration)
                        for _ in range(num_frames):
                            out.write(frame)
                    # Initialize VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'VP80')
                    out = cv2.VideoWriter(output_video_path, fourcc,frame_rate,frame_size)
                    for i in range(len(a)):
                        if(a[i] in arr):
                            ImageAddress = 'letters/'+a[i]+'.jpg'
                            ImageItself = Image.open(ImageAddress)
                            ImageNumpyFormat = np.asarray(ImageItself)
                            plt.imshow(ImageNumpyFormat)
                            plt.draw()
                            plt.pause(0.8)
                            img = cv2.imread(ImageAddress)
                            if img is not None:
                                # Resize image if needed
                                img = cv2.resize(img, frame_size)
                                write_frame_with_duration(img, frame_duration)
                            # cap.release()

                        else:
                            continue
                    out.release()
                    


        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("unknown error occurred")
        plt.close()

while 1:
  image   = "signlang.png"
  msg="HEARING IMPAIRMENT ASSISTANT"
  choices = ["Live Voice","All Done!"] 
  reply   = buttonbox(msg,image=image,choices=choices)
  if reply ==choices[0]:
        func()
  if reply == choices[1]:
        quit()