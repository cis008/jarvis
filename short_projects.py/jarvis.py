
import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import openai 
import pyaudio 
import webbrowser as wb 
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import smtplib
engine=pyttsx3.init("sapi5")
voices=engine.getProperty("voices")
engine.setProperty('voice',voices[0].id)
def sendemail(to,content):
    server=smtplib.SMTP('smtp.gmail.com',578)
    server.ehlo()
    server.starttls()
    server.login("yourEmail@gmail.com",'your-password-here')
    server.sendmail('yourgmail@gmail.com',to,content)
    server.close()
def speak(audio):
    engine.say(audio)
    engine.runAndWait()   
def wishme():
    hour=int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("good morning sir")
    elif hour>=12 and hour<18:
        speak("good afternoon sir")
    else:
        speak("good evening sir")
def takecommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("listening....")
        r.pause_threshold=1
        audio=r.listen(source)
    try:
        print("reognizing....")
        query=r.recognize_google(audio,language='eng-In')
        print(f"you said:{query}")
    except Exception as e:
     #print(e)
     print("say that again pls....")
     return 'None'
    return query
if __name__=='__main__':
     wishme()
     speak("I am Jarvis ,How may I assist you")
     query=takecommand().lower()
     if 'wikipedia' in query:
         speak('searching wikipedia....')
         query=query.replace("wikipedia","")
         result=wikipedia.summary(query,sentences=2)
         speak("according to wikipedia")
         speak(result)
     elif 'open youtube' in  query : 
         wb.open('youtube.com')
     elif 'open google' in query:
         wb.open("google.com")
     elif 'time ' in query:
         strTime=datetime.datetime.now().strftime("%H:%M:%S")
         speak(f"sir the time is {strTime}")
     elif'open code' in query:
         codepath='"C:\\Users\Archit Jagtap\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"'
         os.startfile(codepath)
     elif'send email' in query:

        try:
             speak("what should I say")
             content=takecommand()
             to="architjagtap13@gmail.com ,archit1201"
             sendemail(to,content)
             speak("email has been sent")
        except Exception as e:
            print(e)
        speak("sorry I am not able to send this Email")
    

        



        
    

    

        

     
     


     

  







