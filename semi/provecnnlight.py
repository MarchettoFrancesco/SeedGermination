# CNNTraining_streaming.py

import os
import io
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models

import torch


# Optional: reduce TF INFO logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Set seeds for reproducibility (optional)
np.random.seed(42)
tf.random.set_seed(42)

# ------------------------------------------------------------
# GPU checks
# ------------------------------------------------------------
# PyTorch check (as you had)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TensorFlow GPU check
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
# Configuration (update paths if needed)
# ------------------------------------------------------------
CSV_PATH = "/home/francescomarchetto/semi/DatasetSemi.csv"  # update if your CSV is elsewhere
IMAGES_FOLDER = "/home/francescomarchetto/semi/segmented_seeds_ordered"  # folder you moved into Ubuntu
MODEL_SAVE_PATH = "/home/francescomarchetto/semi/seed_germination_model256.h5"

TARGET_SIZE = (256, 256)           # (width, height)
FILTER_ZEROS = False               # Keep Time=0 rows
MAX_POST_GERM_TIMEPOINTS = 5       # Label 1 for this many timepoints after germination

# ------------------------------------------------------------
# Step 1: Load and clean the CSV
# ------------------------------------------------------------
def load_and_clean_csv(csv_path=CSV_PATH):
    # Read only the needed columns and handle EU decimals and trailing delimiters
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

    # Ensure numeric types
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
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
# Step 2: Index images and dynamically label (do NOT load images)
# ------------------------------------------------------------
def index_and_label_images(df_labels, images_folder=IMAGES_FOLDER, max_post_germ=MAX_POST_GERM_TIMEPOINTS):
    # Map each seed to its germination time
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
        filename = img_path.stem  # e.g., 'scan1_106_p1_seed11_cnn'
        name = filename.replace('_cnn', '')
        parts = name.split('_')
        if len(parts) != 4:
            unmatched_images.append(filename)
            unmatched_details.append(f"Invalid part count ({len(parts)}) for {filename}")
            continue

        scan_part, mezzora_str, p_part, seed_part = parts
        try:
            scan_num = int(scan_part.replace('scan', ''))
            mezzora = int(mezzora_str)  # ensure same unit as 'Time Germinazione'
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

        # Dynamic labeling logic
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

    # Logging
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
        raise ValueError("No matched/labeled data! Check filename units vs 'Time Germinazione' in CSV.")

    return np.array(file_paths), np.array(labels, dtype=np.int32)


# ------------------------------------------------------------
# Step 3: Streaming dataset (load/resize/normalize on the fly)
# ------------------------------------------------------------
def make_dataset(paths, labels, batch_size=32, target_size=TARGET_SIZE, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        # Resize with padding to keep aspect ratio (height, width)
        img = tf.image.resize_with_pad(img, target_size[1], target_size[0])
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(label, tf.float32)

    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------------------------------------------
# Step 4: Build a lighter CNN (avoids huge Flatten)
# ------------------------------------------------------------
def build_cnn_model(input_shape=(256, 256, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
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
# Step 5: Train and evaluate
# ------------------------------------------------------------
def train_and_evaluate(paths, labels, epochs=10, batch_size=32, model_save_path=MODEL_SAVE_PATH):
    # Split paths/labels instead of image arrays
    X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Build datasets
    input_shape = (TARGET_SIZE[1], TARGET_SIZE[0], 3)
    train_ds = make_dataset(X_train, y_train, batch_size=batch_size, training=True)
    val_ds   = make_dataset(X_val,   y_val,   batch_size=batch_size, training=False)
    test_ds  = make_dataset(X_test,  y_test,  batch_size=batch_size, training=False)

    # Class weighting for imbalance
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print(f"Class weights: {class_weight}")

    # Build and train
    model = build_cnn_model(input_shape=input_shape)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1, class_weight=class_weight)

    # Evaluate
    test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
    y_prob = model.predict(test_ds).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\n📊 Test Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Not Germinated', 'Germinated']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

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

    class_report = classification_report(y_test, y_pred, target_names=['Not Germinated', 'Germinated'])
    conf_matrix = confusion_matrix(y_test, y_pred)

    report_content = f"""
# Model Evaluation Report

## Run
- Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Epochs: {epochs}
- Batch size: {batch_size}
- Target size: {TARGET_SIZE}

## Architecture
{architecture}

## Dataset Summary
- Total labeled images: {len(labels)}
- Train: {len(y_train)} (0: {train_not_germinated}, 1: {train_germinated})
- Val: {len(y_val)} (0: {val_not_germinated}, 1: {val_germinated})
- Test: {len(y_test)} (0: {test_not_germinated}, 1: {test_germinated})
- Class weights: {class_weight}

## Test Metrics
- Accuracy: {test_acc:.4f}
- AUC: {test_auc:.4f}

## Classification Report
{class_report}

## Confusion Matrix
{conf_matrix}
"""
    report_path = "model_evaluation_report.md"
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

    # Train
    model, history = train_and_evaluate(paths, labels, epochs=10, batch_size=32, model_save_path=MODEL_SAVE_PATH)