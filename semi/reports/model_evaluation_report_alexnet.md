
# Seed Germination CNN — Alexnet Model

## Run
- Timestamp: 2025-09-16 19:43:05
- Epochs: 12
- Batch size: 32
- Target size: (227, 227)
- Architecture: alexnet (pre-trained)
- Split: by seed (Scan, Piastra, n_seme)
- Class weight mode: none | class_weight={0: 1.0, 1: 15.0}
- Balanced sampling: True
- Threshold strategy: f1

## Dataset Summary
- Total labeled images: 14195
- Train: 9066 (0: 8768, 1: 298)
- Val: 2360 (0: 2275, 1: 85)
- Test: 2769 (0: 2676, 1: 93)

## Test Metrics (default threshold 0.5)
- Accuracy: 0.9693
- AUC: 0.9680

### Classification Report (0.5)
                precision    recall  f1-score   support

Not Germinated       0.99      0.98      0.98      2676
    Germinated       0.53      0.76      0.63        93

      accuracy                           0.97      2769
     macro avg       0.76      0.87      0.80      2769
  weighted avg       0.98      0.97      0.97      2769


### Confusion Matrix (0.5)
[[2613   63]
 [  22   71]]

## Threshold Tuning
- Chosen threshold: 0.495
- Validation F1 at chosen threshold: 0.507

### Classification Report (tuned)
                precision    recall  f1-score   support

Not Germinated       0.99      0.98      0.98      2676
    Germinated       0.53      0.76      0.63        93

      accuracy                           0.97      2769
     macro avg       0.76      0.87      0.80      2769
  weighted avg       0.98      0.97      0.97      2769


### Confusion Matrix (tuned)
[[2613   63]
 [  22   71]]
