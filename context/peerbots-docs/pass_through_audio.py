"""
This file demoes recording your microphone, transcribing what you say, and then sending that to a Peerbots face.
Make sure you change the username you're sending the message to so you can experience the messages.
"""

import speech_recognition as sr
import send_message as send_message

if __name__ == "__main__":
    # Initialize recognizer
    r = sr.Recognizer()

    # Use microphone as source
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    try:
        # Use Google Speech Recognition to transcribe
        text = r.recognize_google(audio)
        print("You said:", text)
        message = {
            "title": "Recognized Message",
            "speech": text,
            "color": "Light Blue",
            "emotion": "Neutral",
        }
        send_message.send_message("GuestUserName", message)

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")

    except sr.RequestError as e:
        print(
            "Could not request results from Google Speech Recognition service; {0}".format(
                e
            )
        )
