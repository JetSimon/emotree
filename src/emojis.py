import os 
from random import choice, randint
import cv2
EMOJI_PATH = "../img"
emojis = os.listdir(EMOJI_PATH)

already_read_np = {}

def preload(size=(16,16)):
    emoji_images = []
    for i in range(len(emojis)):
        emoji_images.append(get_emoji_np(i, size))
    return emoji_images


def preload_cv(size=(16,16)):
    emoji_images = []
    for i in range(len(emojis)):
        emoji_images.append(get_emoji_cv(i, size))
    return emoji_images


def get_random_emoji_index():
    return randint(0, len(emojis) - 1)

def get_random_emoji_cv(size=(16,16)):
    return get_emoji_cv(get_random_emoji_index(), size)

def get_random_emoji_np(size=(16,16)):
    return cv2.cvtColor(get_random_emoji_cv(size=size), cv2.COLOR_BGR2RGBA)

def get_emoji_cv(index, size=(16,16)):
    img = cv2.imread(os.path.join(EMOJI_PATH, emojis[index]), cv2.IMREAD_UNCHANGED)
    return cv2.resize(img, size)

def get_emoji_np(index, size=(16,16)):

    if size == (16, 16) and index in already_read_np:
        return already_read_np[index]

    img = cv2.cvtColor(get_emoji_cv(index, size), cv2.COLOR_BGR2RGBA)

    if size == (16, 16):
        already_read_np[index] = img

    return img