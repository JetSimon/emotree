import emojis
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SIZE = 1024
EMOJI_SIZE = 16
OVERLAP = 1
INPUT_PATH = "../smiling.png"

emoji_images = emojis.preload((EMOJI_SIZE, EMOJI_SIZE))

input_image = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(INPUT_PATH), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA), (SIZE, SIZE))

image = np.zeros_like(input_image)
h, w, c = image.shape

def score_patch(im1, im2):
    return np.linalg.norm(im1[:,:,:3] / 255 - im2[:,:,:3] / 255)

def task(y, x, input_image, output_image):
    chunk = input_image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :]
    chunk_avg = np.average(chunk / 255, axis=(0,1))
    best_chunk = None
    best_score = math.inf
    th, tw, tc = chunk.shape
    for emoji in emoji_images:
        emoji_chunk = emoji[:th, :tw, :tc]
        score = score_patch(chunk, emoji_chunk)

        if score < best_score:
            best_score = score
            best_chunk = emoji_chunk
    
    output_image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :] = best_chunk

coords = []
for y in tqdm(range(0, h, EMOJI_SIZE)):
    for x in tqdm(range(0, w, EMOJI_SIZE)):
        task(y, x, input_image, image)
        coords.append((y,x))

"""with tqdm(total=len(coords)) as pbar:
    with ThreadPoolExecutor() as ex:
        futures = [ex.submit(task, y, x, input_image, image) for y, x in coords]
        for future in as_completed(futures):
            result = future.result()
            pbar.update(1)"""

plt.imshow(image)
plt.savefig("../output/naive.png")
plt.show() 

