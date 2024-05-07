import emojis
import numpy as np
from matplotlib import pyplot as plt
import math
from numpy.random import choice

EMOJI_SIZE = 16
OVERLAP = 1

class Emogent():
    def __init__(self, emojis, shape=(512, 512, 4)):
        self.emojis = emojis
        self.shape = shape

    def get_image(self):
        index = 0
        image = np.zeros(self.shape)
        h, w, c = self.shape
        for y in range(0, h, EMOJI_SIZE // OVERLAP):
            for x in range(0, w, EMOJI_SIZE // OVERLAP):
                chunk = image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :]
                th, tw, tc = chunk.shape
                image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :] = emojis.get_emoji_np(self.emojis[index], size=(EMOJI_SIZE, EMOJI_SIZE))[:th, :tw, :tc]
                index += 1
        return image / 255

    def show(self):
        plt.imshow(self.get_image())
        plt.show()

    def save_image(self, name):
        plt.imshow(self.get_image())
        plt.savefig(name)
        
def breed_emogents(a, b, mut_chance=0.02):
    emoji_list = []
    a_chance = (1.0 - mut_chance) / 2
    b_chance = (1.0 - mut_chance - a_chance)
    
    for i in range(len(a.emojis)):
        roll = choice([a.emojis[i], b.emojis[i], emojis.get_random_emoji_index()], 1, [a_chance, b_chance, mut_chance])[0]
        emoji_list.append(roll)

    return Emogent(np.array(emoji_list))

def create_emogent(image_size=512):
    emojis_needed = math.floor(image_size * image_size / EMOJI_SIZE / EMOJI_SIZE)
    emoji_list = []
    for n in range(emojis_needed):
        emoji_list.append(emojis.get_random_emoji_index())
    return Emogent(np.array(emoji_list))

def create_static_emogent(image_size=512):
    emojis_needed = math.floor(image_size * image_size / EMOJI_SIZE / EMOJI_SIZE)
    emoji_list = []
    e = emojis.get_random_emoji_index()
    for n in range(emojis_needed):
        emoji_list.append(e)
    return Emogent(np.array(emoji_list))

def score_emogent(emo, img):
    e_img = emo.get_image()
    assert(e_img.shape == img.shape)
    
    h,w,c = img.shape

    score = 0

    for y in range(h):
        for x in range(w):
            a,b,c,_ = e_img[y, x, :]
            d,e,f,_ = img[y, x, :]
            score += math.sqrt((a - d)**2 + (b - e)**2 + (c - f)**2)

    return score
            