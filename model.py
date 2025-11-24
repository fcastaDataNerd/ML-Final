import os
from pathlib import Path
import random
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm

CARDS_DIR = Path("/content/SET Dataset")
OUT_DIR = Path("/content/SET_synth_clean")

N_IMAGES = 200
VAL_SPLIT = 0.15

IMG_W, IMG_H = 1800, 1200
GRID_ROWS, GRID_COLS = 3, 4
CARD_W, CARD_H = 350, 300     # FIXED HEIGHT
PADDING_X, PADDING_Y = 80, 60 # FIXED VERTICAL PADDING

random.seed(42)

# Create train/val folders
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

cards = [p for p in CARDS_DIR.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")]

def dark_background(w, h):
    base = np.ones((h, w, 3), dtype=np.uint8) * np.random.randint(30, 60)
    noise = (np.random.randn(h, w, 3) * 5).astype(np.int16)
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))

def add_variations(card):
    if random.random() < 0.5:
        card = ImageEnhance.Brightness(card).enhance(random.uniform(0.95, 1.05))
    if random.random() < 0.5:
        card = ImageEnhance.Contrast(card).enhance(random.uniform(0.95, 1.05))
    return card

def yolo_format(xmin, ymin, xmax, ymax, W, H):
    xc = (xmin + xmax) / (2 * W)
    yc = (ymin + ymax) / (2 * H)
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return xc, yc, w, h

for idx in tqdm(range(N_IMAGES)):
    split = "val" if idx < N_IMAGES * VAL_SPLIT else "train"

    canvas = dark_background(IMG_W, IMG_H)
    labels = []

    selected_cards = random.sample(cards, GRID_ROWS * GRID_COLS)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):

            img_path = selected_cards[r * GRID_COLS + c]
            card = Image.open(img_path).convert("RGB").resize((CARD_W, CARD_H))
            card = add_variations(card)

            x = PADDING_X + c * (CARD_W + PADDING_X)
            y = PADDING_Y + r * (CARD_H + PADDING_Y)

            # Safe bounds check (final protection)
            if x + CARD_W > IMG_W or y + CARD_H > IMG_H:
                continue

            canvas.paste(card, (x, y))

            xmin, ymin = x, y
            xmax, ymax = x + CARD_W, y + CARD_H

            xc, yc, w, h = yolo_format(xmin, ymin, xmax, ymax, IMG_W, IMG_H)

            # ensure all values in [0,1]
            if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                labels.append(f"0 {xc} {yc} {w} {h}")

    # save out
    img_out = OUT_DIR / f"images/{split}/set_{idx}.jpg"
    lbl_out = OUT_DIR / f"labels/{split}/set_{idx}.txt"

    canvas.save(img_out)
    with open(lbl_out, "w") as f:
        f.write("\n".join(labels))


from IPython.display import Image
Image(filename='/content/runs/detect/predict/Screenshot 2025-11-24 101807.jpg')
