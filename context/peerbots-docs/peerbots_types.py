from pydantic import BaseModel
from enum import Enum


class PeerbotsColor(str, Enum):
    light_blue = "Light Blue"
    blue = "Blue"
    green = "Green"
    red = "Red"
    purple = "Purple"
    pink = "Pink"
    yellow = "Yellow"
    orange = "Orange"
    grey = "Grey"
    black = "Black"
    white = "White"

    def __str__(self) -> str:
        return self.value


HexColors = {
    "Light Blue": "#020307",
    "Blue": "#020307",
    "Green": "#020307",
    "Red": "#020307",
    "Purple": "#020307",
    "Pink": "#020307",
    "Yellow": "#020307",
    "Orange": "#020307",
    "Grey": "#020307",
    "Black": "#FDFEFE",
    "White": "#020307",
}


def hexToRGB(hexColor: str):
    h = hexColor.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


class PeerbotsEmotion(str, Enum):
    neutral = "Neutral"
    surprised = "Surprised"
    happy = "Happy"
    sad = "Sad"
    concerned = "Concerned"
    sleepy = "Sleepy"

    def __str__(self) -> str:
        return self.value


class PeerbotsMessage(BaseModel):
    title: str
    speech: str
    color: PeerbotsColor
    emotion: PeerbotsEmotion
    metadata: dict | None = None
