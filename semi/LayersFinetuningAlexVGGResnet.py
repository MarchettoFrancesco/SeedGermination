# cnn_training_with_architectures_pytorch.py

import os
import io
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# GPU checks
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# Configuration (update paths)
# ------------------------------------------------------------
CSV_PATH = "/home/francescomarchetto/semi/DatasetSemi.csv"
IMAGES_FOLDER = "/home/francescomarchetto/semi/segmented_seeds_ordered"

# Configurazione per l'architettura
ARCHITECTURE = 'resnet50'  # 'custom', 'alexnet', 'vgg16', 'resnet50'
PRETRAINED = True  # Usa pesi pre-trained dove possibile

# Paths dinamici basati su ARCHITECTURE per evitare sovrascritture
MODEL_SAVE_PATH = f"/home/francescomarchetto/semi/seed_germination_model_Unfreeze{ARCHITECTURE}.pth"
REPORT_PATH = f"model_evaluation_report_Unfreeze{ARCHITECTURE}.md"

FILTER_ZEROS = False
MAX_POST_GERM_TIMEPOINTS = 5

# Training
EPOCHS = 12
FINE_TUNE_EPOCHS = 5  # Epochs per fine-tuning (setta a 0 per disabilitare)
BATCH_SIZE = 32

# Class weighting or balanced sampling - Fix: Attivato balanced sampling
CLASS_WEIGHT_MODE = "none"
NEG_WEIGHT = 1.0
POS_WEIGHT = 15.0  # Fix: Aumentato

USE_BALANCED_SAMPLING = True

# Threshold tuning
THRESHOLD_STRATEGY = "f1"
TARGET_PRECISION = 0.60

# Data augmentation (mild)
USE_AUGMENT = True

# Cartelle per history e plot
HISTORY_DIR = "histories"
PLOTS_DIR = "plots"

# ------------------------------------------------------------
# Funzioni per Plot (Loss, Accuracy, AUC) e Salvataggio
# ------------------------------------------------------------
def plot_history(history, architecture='custom', save_path=None):
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title(f'Loss - {architecture.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy e AUC (combinati in un subplot)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.plot(epochs, history['auc'], label='Training AUC', linestyle='--')
    plt.plot(epochs, history['val_auc'], label='Validation AUC', linestyle='--')
    plt.title(f'Accuracy and AUC - {architecture.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Graph saved to '{save_path}'")
    plt.close()  # Chiude la figura per evitare display multipli

def plot_comparison(histories_dict, metric='loss', save_path=None):
    plt.figure(figsize=(10, 6))
    for arch, hist in histories_dict.items():
        epochs = range(1, len(hist[metric]) + 1)
        plt.plot(epochs, hist[metric], label=f'{arch.capitalize()} - {metric}')
    
    plt.title(f'Comparison of {metric.capitalize()} Across Models')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Comparison graph saved to '{save_path}'")
    plt.close()

# ------------------------------------------------------------
# Funzione per Threshold Tuning
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
        filename = img_path.stem
        name = filename.replace('_cnn', '')
        parts = name.split('_')
        if len(parts) != 4:
            unmatched_images.append(filename)
            unmatched_details.append(f"Invalid part count ({len(parts)}) for {filename}")
            continue

        scan_part, mezzora_str, p_part, seed_part = parts
        try:
            scan_num = int(scan_part.replace('scan', ''))
            mezzora = int(mezzora_str)
            piastra = int(p_part.replace('p', ''))
            seed_num = int(seed_part.replace('seed', ''))
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

        if germ_time == 0:
            label = 0
        elif mezzora < germ_time:
            label = 0
        elif mezzora <= germ_time + max_post_germ:
            label = 1
        else:
            skipped_grown += 1
            continue

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
# Utilities for splitting by seed
# ------------------------------------------------------------
def parse_seed_key(path_str):
    name = Path(path_str).stem.replace('_cnn', '')
    s, t, piastra, seed = name.split('_')
    scan = int(s.replace('scan', ''))
    piastra = int(piastra.replace('p', ''))
    seed_id = int(seed.replace('seed', ''))
    return (scan, piastra, seed_id)

def split_by_seed(paths, labels, test_size=0.2, val_size=0.2, random_state=42):
    keys = [parse_seed_key(p) for p in paths]
    seeds = sorted(set(keys))
    idx_map = defaultdict(list)
    for i, k in enumerate(keys):
        idx_map[k].append(i)

    seed_to_label = {s: int(np.max(labels[idx_map[s]])) for s in seeds}
    seeds_y = np.array([seed_to_label[s] for s in seeds], dtype=np.int32)

    seeds_train, seeds_test = train_test_split(
        seeds, test_size=test_size, random_state=random_state, stratify=seeds_y
    )
    train_y_for_strat = np.array([seed_to_label[s] for s in seeds_train], dtype=np.int32)
    seeds_train, seeds_val = train_test_split(
        seeds_train, test_size=val_size, random_state=random_state, stratify=train_y_for_strat
    )

    train_set, val_set, test_set = set(seeds_train), set(seeds_val), set(seeds_test)
    train_mask = np.array([k in train_set for k in keys])
    val_mask = np.array([k in val_set for k in keys])
    test_mask = np.array([k in test_set for k in keys])

    return (paths[train_mask], labels[train_mask],
            paths[val_mask], labels[val_mask],
            paths[test_mask], labels[test_mask])

# ------------------------------------------------------------
# Custom Dataset for PyTorch
# ------------------------------------------------------------
class SeedDataset(Dataset):
    def __init__(self, paths, labels, target_size, augment=False):
        self.paths = paths
        self.labels = labels
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard per ImageNet
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(contrast=(0.9, 1.1))
        ]) if augment else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.aug_transform:
            img = self.aug_transform(img)
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

# ------------------------------------------------------------
# Build Model in PyTorch
# ------------------------------------------------------------
def build_model(architecture='custom', input_shape=(3, 256, 256), pretrained=True):
    if architecture == 'custom':
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[1]//4) * (input_shape[2]//4), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    elif architecture == 'alexnet':
        model = models.alexnet(weights='AlexNet_Weights.DEFAULT' if pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    elif architecture == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.DEFAULT' if pretrained else None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    elif architecture == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT' if pretrained else None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    else:
        raise ValueError(f"Architettura non supportata: {architecture}")

    # Inizialmente, freeze tutti i layer eccetto l'head (per fase 1)
    if pretrained and architecture != 'custom':
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze head (classifier/fc)
        head = model.classifier if architecture in ['alexnet', 'vgg16'] else model.fc
        for param in head.parameters():
            param.requires_grad = True

    return model.to(device)

# ------------------------------------------------------------
# Train and Evaluate in PyTorch
# ------------------------------------------------------------
def train_and_evaluate(paths, labels, epochs=EPOCHS, fine_tune_epochs=FINE_TUNE_EPOCHS, batch_size=BATCH_SIZE, model_save_path=MODEL_SAVE_PATH, report_path=REPORT_PATH):
    # Imposta target_size dinamicamente
    if ARCHITECTURE == 'custom':
        target_size = (256, 256)
    elif ARCHITECTURE == 'alexnet':
        target_size = (227, 227)
    elif ARCHITECTURE in ['vgg16', 'resnet50']:
        target_size = (224, 224)
    else:
        raise ValueError(f"Architettura non supportata per target_size: {ARCHITECTURE}")

    print(f"Using target_size: {target_size} for architecture: {ARCHITECTURE}")

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_seed(paths, labels, test_size=0.2, val_size=0.2, random_state=42)

    print(f"\nSplit by seed:")
    print(f"- Train: {len(y_train)} (pos={int(np.sum(y_train))}, neg={int(len(y_train) - np.sum(y_train))})")
    print(f"- Val  : {len(y_val)} (pos={int(np.sum(y_val))}, neg={int(len(y_val) - np.sum(y_val))})")
    print(f"- Test : {len(y_test)} (pos={int(np.sum(y_test))}, neg={int(len(y_test) - np.sum(y_test))})")

    # Datasets
    train_dataset = SeedDataset(X_train, y_train, target_size, augment=USE_AUGMENT)
    val_dataset = SeedDataset(X_val, y_val, target_size, augment=False)
    test_dataset = SeedDataset(X_test, y_test, target_size, augment=False)

    # Balanced sampling (usando WeightedRandomSampler)
    if USE_BALANCED_SAMPLING:
        class_counts = np.bincount(y_train)
        weights = 1.0 / class_counts[y_train]
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = build_model(architecture=ARCHITECTURE, input_shape=(3, target_size[0], target_size[1]), pretrained=PRETRAINED)

    # Optimizer e Loss - Fix: LR basso
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)  # Solo parametri trainable
    criterion = nn.BCELoss()

    # Training loop con "callbacks" manuali - Fix: Patience aumentata
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'auc': [], 'val_auc': []}
    best_val_auc = 0
    patience = 6
    patience_counter = 0
    lr_patience = 2
    lr_counter = 0
    factor = 0.2

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_probs = []
        train_labels = []
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == lbls).sum().item()
            train_total += lbls.size(0)
            train_probs.extend(outputs.detach().cpu().numpy())
            train_labels.extend(lbls.cpu().numpy())

        train_loss /= train_total
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else 0.5

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs).squeeze()
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == lbls).sum().item()
                val_total += lbls.size(0)
                val_probs.extend(outputs.cpu().numpy())
                val_labels.extend(lbls.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Early Stopping e Reduce LR
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            lr_counter = 0
        else:
            patience_counter += 1
            lr_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
            if lr_counter >= lr_patience:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor
                print(f"Reduced LR to {optimizer.param_groups[0]['lr']}")
                lr_counter = 0

    # Fase 2: Fine-Tuning (unfreeze graduale)
    if PRETRAINED and ARCHITECTURE != 'custom' and FINE_TUNE_EPOCHS > 0:
        print("Starting fine-tuning phase...")
        # Unfreeze ultimi layer (graduale per architettura)
        if ARCHITECTURE == 'alexnet':
            for param in model.features[-3:].parameters():  # Unfreeze ultimi 3 conv blocks
                param.requires_grad = True
        elif ARCHITECTURE == 'vgg16':
            for param in model.features[-4:].parameters():  # Unfreeze ultimi 4 layer
                param.requires_grad = True
        elif ARCHITECTURE == 'resnet50':
            for param in model.layer4.parameters():  # Unfreeze ultimo residual block
                param.requires_grad = True

        # Optimizer per fine-tuning (solo parametri requires_grad=True, LR basso)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

        # Continua training
        for epoch in range(FINE_TUNE_EPOCHS):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_probs = []
            train_labels = []
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                outputs = model(imgs).squeeze()
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
                preds = (outputs > 0.5).float()
                train_correct += (preds == lbls).sum().item()
                train_total += lbls.size(0)
                train_probs.extend(outputs.detach().cpu().numpy())
                train_labels.extend(lbls.cpu().numpy())

            train_loss /= train_total
            train_acc = train_correct / train_total
            train_auc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else 0.5

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    outputs = model(imgs).squeeze()
                    loss = criterion(outputs, lbls)
                    val_loss += loss.item() * imgs.size(0)
                    preds = (outputs > 0.5).float()
                    val_correct += (preds == lbls).sum().item()
                    val_total += lbls.size(0)
                    val_probs.extend(outputs.cpu().numpy())
                    val_labels.extend(lbls.cpu().numpy())

            val_loss /= val_total
            val_acc = val_correct / val_total
            val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5

            print(f"Fine-Tune Epoch {epoch+1}/{FINE_TUNE_EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['auc'].append(train_auc)
            history['val_auc'].append(val_auc)

    # Crea cartelle
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Salva history
    history_save_path = os.path.join(HISTORY_DIR, f"history_{ARCHITECTURE}.pkl")
    with open(history_save_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"📊 History saved to '{history_save_path}'")

    # Plotta e salva grafici
    plot_save_path = os.path.join(PLOTS_DIR, f"plots_unfreeze_{ARCHITECTURE}.png")
    plot_history(history, architecture=ARCHITECTURE, save_path=plot_save_path)

    # Evaluate on test
    model.eval()
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            test_probs.extend(outputs.cpu().numpy())
            test_labels.extend(lbls.numpy())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    test_preds = (test_probs > 0.5).astype(int)

    test_acc = (test_preds == test_labels).mean()
    test_auc = roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else 0.5

    report_default = classification_report(test_labels, test_preds, target_names=['Not Germinated', 'Germinated'])
    cm_default = confusion_matrix(test_labels, test_preds)
    print(f"\nDefault threshold 0.5 — Test Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print("Report (0.5):\n", report_default)
    print("Confusion (0.5):\n", cm_default)

    # Threshold tuning on val
    val_probs = []
    val_labels = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            val_probs.extend(outputs.cpu().numpy())
            val_labels.extend(lbls.numpy())

    best_thresh, best_stat = pick_threshold(np.array(val_labels), np.array(val_probs))

    test_preds_thresh = (test_probs > best_thresh).astype(int)
    report_thresh = classification_report(test_labels, test_preds_thresh, target_names=['Not Germinated', 'Germinated'])
    cm_thresh = confusion_matrix(test_labels, test_preds_thresh)

    if THRESHOLD_STRATEGY == "f1":
        print(f"\nBest F1 threshold on val: {best_thresh:.3f} | Val-F1: {best_stat:.3f}")

    print("Report (tuned):\n", report_thresh)
    print("Confusion (tuned):\n", cm_thresh)

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved to {model_save_path}")

    # Save report
    train_not_germinated = int(np.sum(y_train == 0))
    train_germinated = int(np.sum(y_train == 1))
    val_not_germinated = int(np.sum(y_val == 0))
    val_germinated = int(np.sum(y_val == 1))
    test_not_germinated = int(np.sum(y_test == 0))
    test_germinated = int(np.sum(y_test == 1))

    report_content = f"""
# Seed Germination CNN — {ARCHITECTURE.capitalize()} Model

## Run
- Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Epochs: {epochs}
- Fine-Tune Epochs: {FINE_TUNE_EPOCHS}
- Batch size: {batch_size}
- Target size: {target_size}
- Architecture: {ARCHITECTURE} {'(pre-trained)' if PRETRAINED else ''}
- Split: by seed (Scan, Piastra, n_seme)
- Class weight mode: {CLASS_WEIGHT_MODE} | class_weight={{0: {NEG_WEIGHT}, 1: {POS_WEIGHT}}}
- Balanced sampling: {USE_BALANCED_SAMPLING}
- Threshold strategy: {THRESHOLD_STRATEGY}{" | target precision: " + str(TARGET_PRECISION) if THRESHOLD_STRATEGY=="precision_at" else ""}

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
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df_labels = load_and_clean_csv()
    paths, labels = index_and_label_images(df_labels, images_folder=IMAGES_FOLDER, max_post_germ=MAX_POST_GERM_TIMEPOINTS)
    print(f"✅ Prepared {len(paths)} samples (germinated: {int(np.sum(labels))}, not germinated: {int(len(labels) - np.sum(labels))})")

    model, history = train_and_evaluate(paths, labels, epochs=EPOCHS, fine_tune_epochs=FINE_TUNE_EPOCHS, batch_size=BATCH_SIZE, model_save_path=MODEL_SAVE_PATH, report_path=REPORT_PATH)

    # Opzionale: Confronta multiple history (decommenta dopo aver runnato per diversi modelli)
    # histories = {}
    # for arch in ['alexnet', 'vgg16', 'resnet50']:
    #     history_path = os.path.join(HISTORY_DIR, f"history_{arch}.pkl")
    #     if os.path.exists(history_path):
    #         with open(history_path, 'rb') as f:
    #             histories[arch] = pickle.load(f)
    # if histories:
    #     comparison_save_path = os.path.join(PLOTS_DIR, "comparison_val_loss.png")
    #     plot_comparison(histories, metric='val_loss', save_path=comparison_save_path)
    #     # Ripeti per 'val_auc', etc.