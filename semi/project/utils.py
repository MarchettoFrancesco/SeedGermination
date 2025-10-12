import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np
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

def pick_threshold(y_true_val, y_prob_val, strategy, target_precision):
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