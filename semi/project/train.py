import os
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import pickle
import weave
import wandb

from config import *  # Import configs
from data_processing import load_and_clean_csv, index_and_label_images, split_by_seed, SeedDataset
from utils import plot_history, plot_comparison, pick_threshold
from model import build_model

weave.init('marchetto_francesco/SeedGermination') 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# WandB config (log hyperparameters)
config = {
    "architecture": ARCHITECTURE,
    "pretrained": PRETRAINED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": INITIAL_LR,
    "max_post_germ": MAX_POST_GERM_TIMEPOINTS,
    "threshold_strategy": THRESHOLD_STRATEGY,
    "target_precision": TARGET_PRECISION,
    "use_augment": USE_AUGMENT,
}

# WandB init (start tracking)
run = wandb.init(
    project="SeedGermination",
    name=f"{ARCHITECTURE}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=config
)

# Train and Evaluate in PyTorch
def train_and_evaluate(paths, labels, epochs, batch_size, model_save_path, report_path):
    # Imposta target_size dinamicamente
    if ARCHITECTURE == 'custom':
        target_size = (256, 256)
    elif ARCHITECTURE == 'alexnet':
        target_size = (227, 227)
    elif ARCHITECTURE in ["vgg16", "resnet50", "efficientnet_b0", "convnext_tiny", "mobilenet_v3_small"]:
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

    # Balance training set by undersampling majority class (0) to match number of 1's
    print("\nBalancing training set by undersampling majority class (0)...")
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    n_pos = len(pos_indices)
    if len(neg_indices) > n_pos:
        sampled_neg_indices = np.random.choice(neg_indices, n_pos, replace=False)
    else:
        sampled_neg_indices = neg_indices  # If fewer negatives, take all (unlikely)
    balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
    np.random.shuffle(balanced_indices)  # Shuffle for randomness
    X_train = X_train[balanced_indices]
    y_train = y_train[balanced_indices]
    print(f"- Balanced Train: {len(y_train)} (pos={int(np.sum(y_train))}, neg={int(len(y_train) - np.sum(y_train))})")

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
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
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

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
        })

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

    # Crea cartelle
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Salva history
    history_save_path = os.path.join(HISTORY_DIR, f"history_{ARCHITECTURE}_{variant_str}_{timestamp}.pkl")
    with open(history_save_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"📊 History saved to '{history_save_path}'")

    # Plotta e salva grafici
    plot_save_path = os.path.join(PLOTS_DIR, f"plots_{ARCHITECTURE}_{variant_str}_{timestamp}.png")
    plot_history(history, architecture=ARCHITECTURE, save_path=plot_save_path)

    # Log plot to WandB
    wandb.log({"history_plot": wandb.Image(plot_save_path)})

    # Evaluate on test
    model.eval()
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            test_probs.extend(outputs.cpu().numpy().tolist())
            test_labels.extend(lbls.numpy().tolist())

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

    # Log default metrics to WandB
    wandb.log({
        "test_acc_default": test_acc,
        "test_auc_default": test_auc,
    })
    wandb.log({"confusion_matrix_default": wandb.plot.confusion_matrix(probs=None, y_true=test_labels, preds=test_preds, class_names=['Not Germinated', 'Germinated'])})

    # Threshold tuning on val
   
    val_probs = []
    val_labels = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            val_probs.extend(outputs.cpu().numpy().tolist())
            val_labels.extend(lbls.numpy().tolist())

    # Call with all required arguments (strategy and target_precision from config)
    best_thresh, best_stat = pick_threshold(np.array(val_labels), np.array(val_probs), strategy=THRESHOLD_STRATEGY, target_precision=TARGET_PRECISION)

    test_preds_thresh = (test_probs > best_thresh).astype(int)
    report_thresh = classification_report(test_labels, test_preds_thresh, target_names=['Not Germinated', 'Germinated'])
    cm_thresh = confusion_matrix(test_labels, test_preds_thresh)

    if THRESHOLD_STRATEGY == "f1":
        print(f"\nBest F1 threshold on val: {best_thresh:.3f} | Val-F1: {best_stat:.3f}")

    print("Report (tuned):\n", report_thresh)
    print("Confusion (tuned):\n", cm_thresh)

    # Log tuned metrics to WandB
    wandb.log({
        "best_threshold": best_thresh,
        "val_f1": best_stat if THRESHOLD_STRATEGY == "f1" else None,
    })
    wandb.log({"confusion_matrix_tuned": wandb.plot.confusion_matrix(probs=None, y_true=test_labels, preds=test_preds_thresh, class_names=['Not Germinated', 'Germinated'])})
        # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved to {model_save_path}")

    # Log model as artifact to WandB
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_save_path)
    run.log_artifact(artifact)

    # Save report
    train_not_germinated = int(np.sum(y_train == 0))
    train_germinated = int(np.sum(y_train == 1))
    val_not_germinated = int(np.sum(y_val == 0))
    val_germinated = int(np.sum(y_val == 1))
    test_not_germinated = int(np.sum(y_test == 0))
    test_germinated = int(np.sum(y_test == 1))

    report_content = f"""
# Seed Germination CNN — {ARCHITECTURE.capitalize()} Model (Balanced)

## Run
- Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Epochs: {epochs}
- Batch size: {batch_size}
- Target size: {target_size}
- Architecture: {ARCHITECTURE} {'(pre-trained)' if PRETRAINED else ''}
- Split: by seed (Scan, Piastra, n_seme)
- Class weight mode: {CLASS_WEIGHT_MODE} | class_weight={{0: {NEG_WEIGHT}, 1: {POS_WEIGHT}}}
- Balanced sampling: {USE_BALANCED_SAMPLING}
- Training data balanced: Yes (undersampled negatives to match positives)
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

    # Log report as artifact to WandB
    artifact = wandb.Artifact('report', type='report')
    artifact.add_file(report_path)
    run.log_artifact(artifact)

    # Finish WandB run
    run.finish()

    return model, history

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df_labels = load_and_clean_csv(CSV_PATH)
    paths, labels = index_and_label_images(df_labels, images_folder=IMAGES_FOLDER, max_post_germ=MAX_POST_GERM_TIMEPOINTS)
    print(f"✅ Prepared {len(paths)} samples (germinated: {int(np.sum(labels))}, not germinated: {int(len(labels) - np.sum(labels))})")

    model, history = train_and_evaluate(paths, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, model_save_path=MODEL_SAVE_PATH, report_path=REPORT_PATH)