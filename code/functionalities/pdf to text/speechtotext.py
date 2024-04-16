import speech_recognition as sr

def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    
    # Load audio file
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    
    try:
        # Recognize speech using Google Speech Recognition
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return "Error in retrieving results from Google Speech Recognition service; {0}".format(e)

def save_text_to_file(text, filename):
    with open(filename, "w") as file:
        file.write(text)
    

filename = "textt.txt"
# Provide the path to your audio file
audio_file_path = "harvard.wav"
text = audio_to_text(audio_file_path)
save_text_to_file(text, filename)

