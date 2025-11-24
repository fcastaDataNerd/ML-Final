#Part 1
import os
from pathlib import Path
import random
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm
import csv

# ================================
# CONFIGURE PATHS
# ================================
CARDS_DIR = Path(r"C:\Users\franc\Downloads\SET Dataset\SET Dataset") #original 81 card dataset
OUT_DIR   = Path(r"C:\Users\franc\Downloads\SET_synth_clean")

# ================================
# GENERATION SETTINGS
# ================================
N_IMAGES = 200          # Number of 3×4 synthetic boards
VAL_SPLIT = 0.15

IMG_W, IMG_H = 1800, 1200
GRID_ROWS, GRID_COLS = 3, 4
CARD_W, CARD_H = 350, 300
PADDING_X, PADDING_Y = 80, 60

random.seed(42)

# ================================
# CREATE OUTPUT FOLDERS
# ================================
for sub in ["images/train", "images/val"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

cards = [p for p in CARDS_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]

# ================================
# BACKGROUND + AUGMENTATION HELPERS
# ================================
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

# ================================
# NEW ➤ SAVE METADATA FILE
# ================================
meta_file = OUT_DIR / "board_metadata.csv"
with open(meta_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["board_filename"] + [f"slot_{i}" for i in range(1,13)])

# ================================
# GENERATE SYNTHETIC 12 CARD GRIDS
# ================================
for idx in tqdm(range(N_IMAGES)):
    split = "val" if idx < N_IMAGES * VAL_SPLIT else "train"
    canvas = dark_background(IMG_W, IMG_H)

    selected_cards = random.sample(cards, GRID_ROWS * GRID_COLS)

    for i, card_path in enumerate(selected_cards):
        r, c = divmod(i, GRID_COLS)
        x = PADDING_X + c * (CARD_W + PADDING_X)
        y = PADDING_Y + r * (CARD_H + PADDING_Y)

        card = Image.open(card_path).resize((CARD_W, CARD_H)).convert("RGB")
        card = add_variations(card)
        canvas.paste(card, (x, y))

    filename = f"set_{idx}.jpg"
    out_file = OUT_DIR / f"images/{split}/{filename}"
    canvas.save(out_file)

    # SAVE THE FILEPATHS OF 12 CARDS
    with open(meta_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([filename] + [p.name for p in selected_cards])

print("\n🎉 Synthetic boards + metadata generated successfully!")
print("📌 Metadata file saved at:", meta_file)


#Part 2

import cv2
import pandas as pd
from pathlib import Path
import os

# ========= PATHS =========
OUT_DIR = Path(r"C:\Users\franc\Downloads\SET_synth_clean")
IMG_DIR = OUT_DIR / "images"
META_FILE = OUT_DIR / "board_metadata.csv"
CROP_DIR = OUT_DIR / "cropped_cards"

# ========= GRID SETTINGS (must match generator!) =========
IMG_W, IMG_H = 1800, 1200
GRID_ROWS, GRID_COLS = 3, 4
CARD_W, CARD_H = 350, 300
PADDING_X, PADDING_Y = 80, 60

# ========= CREATE OUTPUT FOLDER =========
CROP_DIR.mkdir(exist_ok=True)

# ========= LOAD METADATA =========
df = pd.read_csv(META_FILE)

# ========= LISTS TO BUILD LABELS =========
crop_filenames = []
labels = []

print("\n🔎 Cropping & Labeling Cards...\n")

# ========= LOOP OVER BOARDS =========
for idx, row in df.iterrows():
    board_file = row['board_filename']
    slots = row[1:].values  # slot_1 ... slot_12 filenames

    # find whether it's train or val
    if (IMG_DIR / "train" / board_file).exists():
        board_path = IMG_DIR / "train" / board_file
    else:
        board_path = IMG_DIR / "val" / board_file

    full = cv2.imread(str(board_path))
    full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)

    # loop through 3×4 grid
    for i, label in enumerate(slots):
        r, c = divmod(i, GRID_COLS)

        x1 = PADDING_X + c * (CARD_W + PADDING_X)
        y1 = PADDING_Y + r * (CARD_H + PADDING_Y)
        x2 = x1 + CARD_W
        y2 = y1 + CARD_H

        crop = full[y1:y2, x1:x2]
        crop_name = f"{board_file[:-4]}_slot{i+1}.jpg"

        cv2.imwrite(str(CROP_DIR / crop_name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        crop_filenames.append(crop_name)
        labels.append(label)  # the exact filename label

# ========= SAVE LABELS =========
label_df = pd.DataFrame({
    "filename": crop_filenames,
    "label_filename": labels
})
label_df.to_csv(OUT_DIR / "cropped_card_labels.csv", index=False)

print("\n🎉 Cropping complete!")
print("📌 Cropped cards saved to:", CROP_DIR)
print("📄 Label mapping saved to:", OUT_DIR / "cropped_card_labels.csv")


#Part 3
import pandas as pd
from pathlib import Path

OUT_DIR = Path(r"C:\Users\franc\Downloads\SET_synth_clean")
LABEL_CSV = OUT_DIR / "cropped_card_labels.csv"

df = pd.read_csv(LABEL_CSV)

# ====== Extract attributes from label_filename ======
def parse_label(name):
    name = name.replace(".jpg", "").replace(".jpeg", "")
    parts = name.split("_")
    return parts  # [Number, Fill, Color, Shape]

df[['number','fill','color','shape']] = df['label_filename'].apply(
    lambda x: pd.Series(parse_label(x))
)

# ====== Save enhanced label file ======
out_csv = OUT_DIR / "cropped_card_labels_parsed.csv"
df.to_csv(out_csv, index=False)

print("\n🎉 Labels parsed and saved!")
print("📌 New CSV saved at:", out_csv)
print(df.head())


#Part 4
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Rescaling
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import img_to_array, load_img
from sklearn.model_selection import train_test_split

# =========================================================
# PATHS
# =========================================================
OUT_DIR = Path(r"C:\Users\franc\Downloads\SET_synth_clean")
CROP_DIR = OUT_DIR / "cropped_cards"
CSV_PATH = OUT_DIR / "cropped_card_labels_parsed.csv"

# =========================================================
# LOAD LABEL CSV
# =========================================================
df = pd.read_csv(CSV_PATH)

print("Loaded labels:", df.shape)

# =========================================================
# MAPPINGS
# =========================================================
number_map = {"One": 0, "Two": 1, "Three": 2}
fill_map   = {"Open": 0, "Shaded": 1, "Solid": 2}
color_map  = {"Green": 0, "Purple": 1, "Red": 2}
shape_map  = {"Diamond": 0, "Oval": 1, "Squiggle": 2}

# numerical encoding
df["y_number"] = df["number"].map(number_map)
df["y_fill"]   = df["fill"].map(fill_map)
df["y_color"]  = df["color"].map(color_map)
df["y_shape"]  = df["shape"].map(shape_map)

# =========================================================
# LOAD IMAGES
# =========================================================
input_dim = 224  # colleague 2’s input size

X = []
y_num = []
y_fill = []
y_color = []
y_shape = []

print("\nLoading images...")

for idx, row in df.iterrows():
    img_path = CROP_DIR / row["filename"]
    img = load_img(img_path, target_size=(input_dim, input_dim))
    arr = img_to_array(img)
    X.append(arr)

    y_num.append(row["y_number"])
    y_fill.append(row["y_fill"])
    y_color.append(row["y_color"])
    y_shape.append(row["y_shape"])

X = np.array(X)
y_num = np.array(y_num)
y_fill = np.array(y_fill)
y_color = np.array(y_color)
y_shape = np.array(y_shape)

print("Image array shape:", X.shape)

# =========================================================
# TRAIN/VAL SPLIT (20% validation)
# =========================================================
X_train, X_val, yn_train, yn_val, yf_train, yf_val, yc_train, yc_val, ys_train, ys_val = \
    train_test_split(
        X, y_num, y_fill, y_color, y_shape, 
        test_size=0.20, random_state=42
    )

# =========================================================
# BUILD MULTI-OUTPUT CNN (colleague 2’s architecture)
# =========================================================
inputs = Input(shape=(input_dim, input_dim, 3))
x = Rescaling(1/255)(inputs)

x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = MaxPool2D()(x)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = MaxPool2D()(x)
x = Conv2D(128, 3, activation='relu', padding='same')(x)
x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)

out_num   = Dense(3, activation='softmax', name="number")(x)
out_fill  = Dense(3, activation='softmax', name="fill")(x)
out_color = Dense(3, activation='softmax', name="color")(x)
out_shape = Dense(3, activation='softmax', name="shape")(x)

model = Model(inputs, [out_num, out_fill, out_color, out_shape])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics={"number": "accuracy", "fill": "accuracy", "color": "accuracy", "shape": "accuracy"},
)

# =========================================================
# TRAIN
# =========================================================
save_path = OUT_DIR / "set_cnn_model.keras"

callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ModelCheckpoint(save_path, save_best_only=True)
]

history = model.fit(
    X_train,
    [yn_train, yf_train, yc_train, ys_train],
    validation_data=(X_val, [yn_val, yf_val, yc_val, ys_val]),
    epochs=25,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\nModel saved to:", save_path)


#Part 5
import cv2
import numpy as np
import random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# ======================================================
#            PATHS + MODEL LOAD
# ======================================================
OUT_DIR = Path(r"C:\Users\franc\Downloads\SET_synth_clean")
IMG_DIR = OUT_DIR / "images"
MODEL_PATH = OUT_DIR / "set_cnn_model.keras"

model = load_model(MODEL_PATH)

# ======================================================
#          LABEL DECODING MAPS
# ======================================================
num_map   = {0:'One',   1:'Two',    2:'Three'}
fill_map  = {0:'Open',  1:'Shaded', 2:'Solid'}
color_map = {0:'Green', 1:'Purple', 2:'Red'}
shape_map = {0:'Diamond', 1:'Oval', 2:'Squiggle'}

# ======================================================
#          CROP A CARD AND CLASSIFY IT
# ======================================================
def classify_card(img):
    resized = cv2.resize(img, (224, 224))
    arr = img_to_array(resized)
    arr = np.expand_dims(arr, axis=0)

    pn, pf, pc, ps = model.predict(arr, verbose=0)

    return {
        "Number": num_map[np.argmax(pn)],
        "Fill":   fill_map[np.argmax(pf)],
        "Color":  color_map[np.argmax(pc)],
        "Shape":  shape_map[np.argmax(ps)]
    }

# ======================================================
#      IDENTIFY + CROP 12 CARDS FROM A RANDOM BOARD
# ======================================================
def test_random_board(split="train"):
    imgs = list((IMG_DIR / split).glob("*.jpg"))
    img_path = random.choice(imgs)

    full = cv2.imread(str(img_path))
    full_rgb = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)

    # Show the board
    plt.figure(figsize=(12, 8))
    plt.imshow(full_rgb)
    plt.axis("off")
    plt.title(f"Random Test Board — {img_path.name}")
    plt.show()

    # Crop parameters (must match generator)
    CARD_W, CARD_H = 350, 300
    PADDING_X, PADDING_Y = 80, 60
    GRID_ROWS, GRID_COLS = 3, 4

    classified_cards = {}
    idx = 1

    # Visualization subplot
    plt.figure(figsize=(8, 6))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1 = PADDING_X + c * (CARD_W + PADDING_X)
            y1 = PADDING_Y + r * (CARD_H + PADDING_Y)
            x2 = x1 + CARD_W
            y2 = y1 + CARD_H

            crop = full_rgb[y1:y2, x1:x2]
            card_key = f"Card{r+1}{c+1}"

            # classify the crop
            classified_cards[card_key] = classify_card(crop)

            # show the crop
            plt.subplot(3, 4, idx)
            plt.imshow(crop); plt.axis("off"); plt.title(card_key)
            idx += 1

    plt.suptitle("Cropped Cards + Positions", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\n🔎 CLASSIFIED CARD DICTIONARY:\n")
    for k, v in classified_cards.items():
        print(k, v)

    return classified_cards

# 🚀 RUN IT!
result = test_random_board("train")
