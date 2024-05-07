import emogent
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from tqdm import tqdm
import math

GEN_SIZE = 25
img = cv2.cvtColor(cv2.imread(os.path.join("../rainbow.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)

print("Creating initial agents...")
generation = []
for i in tqdm(range(GEN_SIZE)):
    generation.append(emogent.create_emogent())

winners_to_pick = math.floor(math.sqrt(GEN_SIZE))

for epoch in range(10):
    print("Scoring agnents...")

    scores = {}
    for emo in tqdm(generation):
        scores[emo] = emogent.score_emogent(emo, img)

    winners = sorted(generation, key=lambda x : scores[x])[:winners_to_pick]

    print(list(map(lambda x : scores[x], winners)))

    print("Saving best images...")

    for i in range(min(3, len(winners))):
        winners[i].save_image(f"../output/epoch_{epoch}_{i+1}_place.png")

    generation = []

    print("Breeding...")

    for i in range(len(winners)):
        for j in range(len(winners)):
            c = emogent.breed_emogents(winners[i], winners[j])
            generation.append(c)

