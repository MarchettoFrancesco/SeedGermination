# cnn_training_flatten_seed_split_threshold.py

import os
import io
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models

import torch

# Reduce TF INFO logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ------------------------------------------------------------
# GPU checks
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {gpus}")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")

# ------------------------------------------------------------
# Configuration (update paths)
# ------------------------------------------------------------
CSV_PATH = "/home/francescomarchetto/semi/DatasetSemi.csv"
IMAGES_FOLDER = "/home/francescomarchetto/semi/segmented_seeds_ordered"

MODEL_SAVE_PATH = "/home/francescomarchetto/semi/seed_germination_model_flattenHeavier.keras"
REPORT_PATH = "model_evaluation_report_flattenHeavier.md"

# Image size: 256x256 → ~31.5M params with Flatten; 128x128 → ~7.4M params
TARGET_SIZE = (256, 256)  # (width, height)

FILTER_ZEROS = False
MAX_POST_GERM_TIMEPOINTS = 5  # label 1 for this many timepoints after germination

# Training
EPOCHS = 12
BATCH_SIZE = 32

# Class weighting or balanced sampling (pick one)
CLASS_WEIGHT_MODE = "manual"  # "manual", "balanced", or "none"
NEG_WEIGHT = 1.0
POS_WEIGHT = 6.0  # try 4–10; larger => more recall, more FPs

USE_BALANCED_SAMPLING = False  # True = 50/50 sampling of pos/neg in train_ds (then set CLASS_WEIGHT_MODE="none")

# Threshold tuning
THRESHOLD_STRATEGY = "f1"  # "f1" or "precision_at"
TARGET_PRECISION = 0.60     # used if THRESHOLD_STRATEGY == "precision_at"

# Data augmentation (mild)
USE_AUGMENT = True

# ------------------------------------------------------------
# Step 1: Load and clean CSV
# ------------------------------------------------------------
def load_and_clean_csv(csv_path=CSV_PATH):
    df = pd.read_csv(
        csv_path,
        sep=';',
        usecols=['Scan', 'Piastra', 'n_seme', 'Time Germinazione'],
        decimal=',',
        engine='python'
    )
    print("Loaded DataFrame head:")
    print(df.head())
    print("Original shape:", df.shape)

    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df[['Scan', 'Piastra', 'n_seme', 'Time Germinazione']] = df[['Scan', 'Piastra', 'n_seme', 'Time Germinazione']].astype('int32')

    if FILTER_ZEROS:
        df = df[df['Time Germinazione'] > 0]
        print("\nFiltered out Time=0 rows.")
    else:
        print("\nKeeping Time=0 rows for dynamic labeling.")

    print("Cleaned DataFrame head:")
    print(df.head())
    print("Shape after cleaning:", df.shape)
    return df

# ------------------------------------------------------------
# Step 2: Index & label images (no image loading)
# ------------------------------------------------------------
def index_and_label_images(df_labels, images_folder=IMAGES_FOLDER, max_post_germ=MAX_POST_GERM_TIMEPOINTS):
    seed_to_germ = {
        (int(r['Scan']), int(r['Piastra']), int(r['n_seme'])): int(r['Time Germinazione'])
        for _, r in df_labels.iterrows()
    }
    print(f"Loaded germination times for {len(seed_to_germ)} unique seeds.")

    image_paths = sorted(Path(images_folder).glob('*_cnn.png'))
    print(f"Found {len(image_paths)} _cnn images in {images_folder}")

    file_paths, labels = [], []
    unmatched_images, unmatched_details = [], []
    matched_count, skipped_grown = 0, 0

    for img_path in image_paths:
        filename = img_path.stem  # 'scan1_106_p1_seed11_cnn'
        name = filename.replace('_cnn', '')
        parts = name.split('_')
        if len(parts) != 4:
            unmatched_images.append(filename)
            unmatched_details.append(f"Invalid part count ({len(parts)}) for {filename}")
            continue

        scan_part, mezzora_str, p_part, seed_part = parts
        try:
            scan_num = int(scan_part.replace('scan', ''))
            mezzora = int(mezzora_str)        # must match 'Time Germinazione' units
            piastra = int(p_part.replace('p', ''))
            seed_num = int(seed_part.replace('seed', ''))  # '01' -> 1 works
        except ValueError as e:
            unmatched_images.append(filename)
            unmatched_details.append(f"Parse error ({str(e)}) for {filename}")
            continue

        key = (scan_num, piastra, seed_num)
        germ_time = seed_to_germ.get(key)
        if germ_time is None:
            unmatched_images.append(filename)
            unmatched_details.append(f"No germination time in CSV for (Scan={scan_num}, Piastra={piastra}, n_seme={seed_num}) from {filename}")
            continue

        # Dynamic labeling
        if germ_time == 0:
            label = 0
        elif mezzora < germ_time:
            label = 0
        elif mezzora <= germ_time + max_post_germ:
            label = 1
        else:
            skipped_grown += 1
            continue  # skip "too grown"

        file_paths.append(str(img_path))
        labels.append(label)
        matched_count += 1

    if matched_count > 0:
        pos = int(np.sum(labels))
        neg = matched_count - pos
        print(f"✅ Matched and labeled {matched_count} images (including dynamic labels).")
        print(f"📊 Label counts — Germinated: {pos}, Not germinated: {neg}")

    if skipped_grown > 0:
        print(f"📊 Skipped {skipped_grown} 'too grown' images (mezzora > germ_time + {max_post_germ}).")

    if unmatched_images:
        print(f"⚠️ {len(unmatched_images)} unmatched _cnn images (e.g., {unmatched_images[:5]})")
        print("Details for first few unmatched:\n" + "\n".join(unmatched_details[:5]))

    if len(file_paths) == 0:
        raise ValueError("No matched/labeled data! Check filename units vs 'Time Germinazione'.")

    return np.array(file_paths), np.array(labels, dtype=np.int32)

# ------------------------------------------------------------
# Utilities for splitting by seed (avoid leakage)
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

def parse_seed_key(path_str):
    name = Path(path_str).stem.replace('_cnn', '')
    s, t, piastra, seed = name.split('_')
    scan = int(s.replace('scan', ''))
    piastra = int(piastra.replace('p', ''))
    seed_id = int(seed.replace('seed', ''))
    return (scan, piastra, seed_id)

def split_by_seed(paths, labels, test_size=0.2, val_size=0.2, random_state=42):
    # Keys per image (list of tuples)
    keys = [parse_seed_key(p) for p in paths]

    # Unique seeds as a list of tuples (do NOT convert to np.array here)
    seeds = sorted(set(keys))

    # Map seed -> indices and seed -> label (1 if seed has any positive frame)
    from collections import defaultdict
    idx_map = defaultdict(list)
    for i, k in enumerate(keys):
        idx_map[k].append(i)

    seed_to_label = {s: int(np.max(labels[idx_map[s]])) for s in seeds}

    # Stratify by seed-level label
    seeds_y = np.array([seed_to_label[s] for s in seeds], dtype=np.int32)

    # Seed-level splits
    seeds_train, seeds_test = train_test_split(
        seeds, test_size=test_size, random_state=random_state, stratify=seeds_y
    )
    train_y_for_strat = np.array([seed_to_label[s] for s in seeds_train], dtype=np.int32)
    seeds_train, seeds_val = train_test_split(
        seeds_train, test_size=val_size, random_state=random_state, stratify=train_y_for_strat
    )

    # Build masks for images that belong to each seed split
    train_set, val_set, test_set = set(seeds_train), set(seeds_val), set(seeds_test)
    train_mask = np.array([k in train_set for k in keys])
    val_mask   = np.array([k in val_set for k in keys])
    test_mask  = np.array([k in test_set for k in keys])

    return (paths[train_mask], labels[train_mask],
            paths[val_mask],   labels[val_mask],
            paths[test_mask],  labels[test_mask])

# ------------------------------------------------------------
# Step 3: Streaming dataset with optional augmentation
# ------------------------------------------------------------
def make_dataset(paths, labels, batch_size=32, target_size=TARGET_SIZE, training=True, augment=USE_AUGMENT):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.image.resize_with_pad(img, target_size[1], target_size[0])  # (H, W)
        img = tf.cast(img, tf.float32) / 255.0
        label = tf.cast(label, tf.float32)
        return img, label

    def _augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        return img, label

    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training and augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def make_balanced_train_dataset(paths, labels, batch_size=32, target_size=TARGET_SIZE, augment=USE_AUGMENT):
    base = tf.data.Dataset.from_tensor_slices((paths, labels))
    pos  = base.filter(lambda p, y: tf.equal(y, 1))
    neg  = base.filter(lambda p, y: tf.equal(y, 0))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.image.resize_with_pad(img, target_size[1], target_size[0])
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(label, tf.float32)

    def _augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    sampled = tf.data.Dataset.sample_from_datasets([pos, neg], weights=[0.5, 0.5], stop_on_empty_dataset=False)
    ds = sampled.shuffle(10000).map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ------------------------------------------------------------
# Step 4: Original Flatten CNN
# ------------------------------------------------------------
def build_cnn_model(input_shape=(256, 256, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    model.summary()
    return model

# ------------------------------------------------------------
# Step 5: Train and evaluate with threshold tuning
# ------------------------------------------------------------
def pick_threshold(y_true_val, y_prob_val, strategy=THRESHOLD_STRATEGY, target_precision=TARGET_PRECISION):
    prec, rec, th = precision_recall_curve(y_true_val, y_prob_val)
    if strategy == "f1":
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = int(np.argmax(f1))
        best_thresh = th[best_idx] if best_idx < len(th) else 0.5
        return float(best_thresh), float(f1[best_idx])
    elif strategy == "precision_at":
        idx = np.where(prec[:-1] >= target_precision)[0]
        if len(idx):
            return float(th[idx[0]]), None
        return 0.5, None
    else:
        return 0.5, None

def train_and_evaluate(paths, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, model_save_path=MODEL_SAVE_PATH, report_path=REPORT_PATH):
    # Seed-level split
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_seed(paths, labels, test_size=0.2, val_size=0.2, random_state=42)

    print(f"\nSplit by seed:")
    print(f"- Train: {len(y_train)} (pos={int(np.sum(y_train))}, neg={int(len(y_train) - np.sum(y_train))})")
    print(f"- Val  : {len(y_val)} (pos={int(np.sum(y_val))}, neg={int(len(y_val) - np.sum(y_val))})")
    print(f"- Test : {len(y_test)} (pos={int(np.sum(y_test))}, neg={int(len(y_test) - np.sum(y_test))})")

    input_shape = (TARGET_SIZE[1], TARGET_SIZE[0], 3)

    # Datasets
    if USE_BALANCED_SAMPLING:
        train_ds = make_balanced_train_dataset(X_train, y_train, batch_size=batch_size, target_size=TARGET_SIZE, augment=USE_AUGMENT)
        class_weight = None
        print("Using balanced sampling for training; class_weight disabled.")
    else:
        train_ds = make_dataset(X_train, y_train, batch_size=batch_size, target_size=TARGET_SIZE, training=True, augment=USE_AUGMENT)
        if CLASS_WEIGHT_MODE == "balanced":
            classes = np.array([0, 1])
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight = {0: float(weights[0]), 1: float(weights[1])}
        elif CLASS_WEIGHT_MODE == "manual":
            class_weight = {0: float(NEG_WEIGHT), 1: float(POS_WEIGHT)}
        else:
            class_weight = None
        print(f"Class weight mode: {CLASS_WEIGHT_MODE} | class_weight={class_weight}")

    val_ds  = make_dataset(X_val,  y_val,  batch_size=batch_size, target_size=TARGET_SIZE, training=False, augment=False)
    test_ds = make_dataset(X_test, y_test, batch_size=batch_size, target_size=TARGET_SIZE, training=False, augment=False)

    # Model
    model = build_cnn_model(input_shape=input_shape)

    # Callbacks
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', patience=2, factor=0.5)
    ]

    # Train
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1, class_weight=class_weight, callbacks=cbs)

    # Evaluate at default threshold 0.5
    test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
    y_prob_test = model.predict(test_ds).ravel()
    y_pred_test = (y_prob_test > 0.5).astype(int)

    report_default = classification_report(y_test, y_pred_test, target_names=['Not Germinated', 'Germinated'])
    cm_default = confusion_matrix(y_test, y_pred_test)
    print(f"\nDefault threshold 0.5 — Test Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print("Report (0.5):\n", report_default)
    print("Confusion (0.5):\n", cm_default)

    # Threshold tuning on validation set
    y_prob_val = model.predict(val_ds).ravel()
    best_thresh, best_stat = pick_threshold(y_val, y_prob_val, strategy=THRESHOLD_STRATEGY, target_precision=TARGET_PRECISION)

    y_pred_thresh = (y_prob_test > best_thresh).astype(int)
    report_thresh = classification_report(y_test, y_pred_thresh, target_names=['Not Germinated', 'Germinated'])
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)

    if THRESHOLD_STRATEGY == "f1":
        print(f"\nBest F1 threshold on val: {best_thresh:.3f} | Val-F1: {best_stat:.3f}")
    elif THRESHOLD_STRATEGY == "precision_at":
        print(f"\nThreshold on val for precision≥{TARGET_PRECISION}: {best_thresh:.3f}")

    print("Report (tuned):\n", report_thresh)
    print("Confusion (tuned):\n", cm_thresh)

    # Save model
    model.save(model_save_path)
    print(f"💾 Model saved to {model_save_path}")

    # Save evaluation report (Markdown)
    summary_io = io.StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
    architecture = summary_io.getvalue()

    train_not_germinated = int(np.sum(y_train == 0))
    train_germinated = int(np.sum(y_train == 1))
    val_not_germinated = int(np.sum(y_val == 0))
    val_germinated = int(np.sum(y_val == 1))
    test_not_germinated = int(np.sum(y_test == 0))
    test_germinated = int(np.sum(y_test == 1))

    report_content = f"""
# Seed Germination CNN — Flatten Model

## Run
- Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Epochs: {epochs}
- Batch size: {batch_size}
- Target size: {TARGET_SIZE}
- Split: by seed (Scan, Piastra, n_seme)
- Class weight mode: {CLASS_WEIGHT_MODE} | class_weight={class_weight}
- Balanced sampling: {USE_BALANCED_SAMPLING}
- Threshold strategy: {THRESHOLD_STRATEGY}{" | target precision: " + str(TARGET_PRECISION) if THRESHOLD_STRATEGY=="precision_at" else ""}

## Architecture
{architecture}

## Dataset Summary
- Total labeled images: {len(labels)}
- Train: {len(y_train)} (0: {train_not_germinated}, 1: {train_germinated})
- Val: {len(y_val)} (0: {val_not_germinated}, 1: {val_germinated})
- Test: {len(y_test)} (0: {test_not_germinated}, 1: {test_germinated})

## Test Metrics (default threshold 0.5)
- Accuracy: {test_acc:.4f}
- AUC: {test_auc:.4f}

### Classification Report (0.5)
{report_default}

### Confusion Matrix (0.5)
{cm_default}

## Threshold Tuning
- Chosen threshold: {best_thresh:.3f}
{"- Validation F1 at chosen threshold: " + f"{best_stat:.3f}" if THRESHOLD_STRATEGY=="f1" else ""}

### Classification Report (tuned)
{report_thresh}

### Confusion Matrix (tuned)
{cm_thresh}
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"📄 Evaluation report saved to '{report_path}'")

    return model, history

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    df_labels = load_and_clean_csv()
    paths, labels = index_and_label_images(df_labels, images_folder=IMAGES_FOLDER, max_post_germ=MAX_POST_GERM_TIMEPOINTS)
    print(f"✅ Prepared {len(paths)} samples (germinated: {int(np.sum(labels))}, not germinated: {int(len(labels) - np.sum(labels))})")

    model, history = train_and_evaluate(paths, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, model_save_path=MODEL_SAVE_PATH, report_path=REPORT_PATH)