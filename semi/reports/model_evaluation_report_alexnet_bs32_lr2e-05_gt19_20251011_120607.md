
# Seed Germination CNN — Alexnet Model (Balanced)

## Run
- Timestamp: 2025-10-11 12:08:39
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
- Total labeled images: 15107
- Train: 1688 (0: 844, 1: 844)
- Val: 2531 (0: 2275, 1: 256)
- Test: 2964 (0: 2676, 1: 288)

## Test Metrics (default threshold 0.5)
- Accuracy: 0.9484
- AUC: 0.9684

### Classification Report (0.5)
                precision    recall  f1-score   support

Not Germinated       0.99      0.95      0.97      2676
    Germinated       0.68      0.89      0.77       288

      accuracy                           0.95      2964
     macro avg       0.83      0.92      0.87      2964
  weighted avg       0.96      0.95      0.95      2964


### Confusion Matrix (0.5)
[[2555  121]
 [  32  256]]

## Threshold Tuning
- Chosen threshold: 0.749
- Validation F1 at chosen threshold: 0.699

### Classification Report (tuned)
                precision    recall  f1-score   support

Not Germinated       0.98      0.97      0.98      2676
    Germinated       0.78      0.85      0.81       288

      accuracy                           0.96      2964
     macro avg       0.88      0.91      0.90      2964
  weighted avg       0.96      0.96      0.96      2964


### Confusion Matrix (tuned)
[[2605   71]
 [  42  246]]
