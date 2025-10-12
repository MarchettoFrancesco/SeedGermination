
# Seed Germination CNN — Alexnet Model (Balanced)

## Run
- Timestamp: 2025-10-11 09:33:02
- Epochs: 12
- Batch size: 32
- Target size: (227, 227)
- Architecture: alexnet (pre-trained)
- Split: by seed (Scan, Piastra, n_seme)
- Class weight mode: none | class_weight={0: 1.0, 1: 15.0}
- Balanced sampling: False
- Training data balanced: Yes (undersampled negatives to match positives)
- Threshold strategy: f1

## Dataset Summary
- Total labeled images: 14870
- Train: 1406 (0: 703, 1: 703)
- Val: 2487 (0: 2275, 1: 212)
- Test: 2912 (0: 2676, 1: 236)

## Test Metrics (default threshold 0.5)
- Accuracy: 0.9289
- AUC: 0.9641

### Classification Report (0.5)
                precision    recall  f1-score   support

Not Germinated       0.99      0.93      0.96      2676
    Germinated       0.54      0.91      0.68       236

      accuracy                           0.93      2912
     macro avg       0.76      0.92      0.82      2912
  weighted avg       0.95      0.93      0.94      2912


### Confusion Matrix (0.5)
[[2490  186]
 [  21  215]]

## Threshold Tuning
- Chosen threshold: 0.854
- Validation F1 at chosen threshold: 0.646

### Classification Report (tuned)
                precision    recall  f1-score   support

Not Germinated       0.98      0.97      0.98      2676
    Germinated       0.70      0.77      0.74       236

      accuracy                           0.96      2912
     macro avg       0.84      0.87      0.86      2912
  weighted avg       0.96      0.96      0.96      2912


### Confusion Matrix (tuned)
[[2599   77]
 [  54  182]]
