import emojis
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn import tree

WIDTH = 1920 // 2
HEIGHT = 1080 // 2

EMOJI_SIZE = 16
OVERLAP = 1
INPUT_PATH = "../smiling.png"

def get_average_color(img):
    return np.average(img[:,:,:3] / 255, axis=(0,1))

emoji_images = emojis.preload_cv((EMOJI_SIZE, EMOJI_SIZE))

print("training tree")
clf = tree.DecisionTreeClassifier()
X = np.array(list(map(get_average_color, emoji_images)))
Y = np.array([n for n in range(0, len(emoji_images))])

clf = clf.fit(X, Y)

def score_patch(im1, im2):
    return np.linalg.norm(im1[:,:,:3] / 255 - im2[:,:,:3] / 255)

def get_avg_chunk_at(coord):
    y, x = coord
    chunk = input_image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :]
    return get_average_color(chunk)

def task(y, x, input_image, output_image):
    chunk = input_image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :]
    """best_chunk = None
    best_score = math.inf
    th, tw, tc = chunk.shape
    
    for emoji in emoji_images:
        emoji_chunk = emoji[:th, :tw, :tc]
        score = score_patch(chunk, emoji_chunk)

        if score < best_score:
            best_score = score
            best_chunk = emoji_chunk
    """

    #output_image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :] = best_chunk

vid = cv2.VideoCapture(0) 

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    input_image = cv2.resize(frame, (WIDTH, HEIGHT))


    image = np.zeros_like(input_image)
    h, w, c = image.shape

    coords = []
    for y in range(0, h, EMOJI_SIZE):
        for x in range(0, w, EMOJI_SIZE):
            coords.append((y,x))

    X = list(map(get_avg_chunk_at, coords))
    Y = clf.predict(X)

    for i in range(len(coords)):
        y, x = coords[i]

        chunk = input_image[y:y+EMOJI_SIZE, x:x+EMOJI_SIZE, :]
        th, tw, tc = chunk.shape

        image[y:y+th, x:x+tw, :3] = emoji_images[Y[i]][:th, :tw, :3]

    h,w,c = frame.shape
    h = h // 4
    w = w //4

    frame = cv2.resize(frame, (w,h))

    image[:h,:w,:3] = frame

    # Display the resulting frame 
    cv2.imshow('frame', image) 
    
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

