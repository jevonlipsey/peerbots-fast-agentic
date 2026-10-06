from picozero import pico_led, Button
import requests
import network
from time import sleep

ssid = 'WIFI_NETWORK_NAME'
password = 'WIFI_NETWORK_PASSWORD'

def connect_to_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    while wlan.isconnected() == False:
        print('Waiting for connection..')
        sleep(1)
    # print(wlan.ifconfig())
    ip = wlan.ifconfig()[0]
    print(f'Connected on {ip}')
    return ip

counter = 0

def main():
    print("Attempting to connect to wifi")
    ip = connect_to_wifi()
    print("Starting program")
    while True:
        counter_button = Button(15)
        counter_button.when_pressed = on_press
        counter_button.when_released = counter_off_press

        welcome_button = Button(14)
        welcome_button.when_pressed = on_press
        welcome_button.when_released = welcome_off_press



def on_press():
    pico_led.on()

def counter_off_press():
    pico_led.off()
    global counter 
    counter+=1

    send_message("USERNAME_OR_EMAIL_OF_FACE", {
        "title": "Counter",
        "speech": f"Someone pushed this button {counter} times today!",
        "color": "Light Blue",
        "emotion": "Neutral",
    })

def welcome_off_press():
    pico_led.off()

    send_message("USERNAME_OR_EMAIL_OF_FACE", {
        "title": "Introduction and backstory",
        "speech": f"Hi, my name is _______, and let me tell you my story. ____________",
        "color": "Light Blue",
        "emotion": "Happy",
    })

url = "https://api.peerbots.org/v1"
api_key = "YOUR_API_KEY"

headers = {"Accept": "application/json", "X-API-KEY": api_key}


def send_message(username: str, message):
    """Send a message to a username or email

    Args:
        username (str): The username or email of the face to message
        message (PeerbotsMessage): The message to send to the face
    """
    req = requests.post(f"{url}/send-message/{username}", headers=headers, json=message)
    print(req.json())


if __name__ == "__main__":
    main()
