import emojis
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

emojis.preload()

EMOJI_SIZE = 16
OVERLAP = 1

input_image = cv2.cvtColor(cv2.imread(os.path.join("../rainbow.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)

image = np.zeros_like(input_image)
h, w, c = image.shape

def score_patch(im1, im2):
    assert(im1.shape == im2.shape)

    h,w,c = im1.shape
    score = 0

    for y in range(h):
        for x in range(w):
            a,b,c,_ = im1[y, x, :]
            d,e,f,_ = im2[y, x, :]
            score += math.sqrt((a - d)**2 + (b - e)**2 + (c - f)**2)

    return score

def task(y, x, image):
    chunk = image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :]
    best_chunk = None
    best_score = math.inf
    th, tw, tc = chunk.shape
    for emoji in emojis.emoji_images:
        emoji_chunk = emoji[:th, :tw, :tc]
        score = score_patch(emoji_chunk, chunk)

        if best_score > score:
            best_score = score
            best_chunk = chunk
    
    image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :] = best_chunk

coords = []

for y in range(0, h, EMOJI_SIZE // OVERLAP):
    for x in range(0, w, EMOJI_SIZE // OVERLAP):
        coords.append((y,x))

print("Done adding coords")

with tqdm(total=len(coords)) as pbar:
    with ThreadPoolExecutor(max_workers=len(coords)) as ex:
        futures = [ex.submit(task, y, x, image) for y,x in coords]
        for future in as_completed(futures):
            result = future.result()
            pbar.update(1)

plt.imshow(image)
plt.show() 
