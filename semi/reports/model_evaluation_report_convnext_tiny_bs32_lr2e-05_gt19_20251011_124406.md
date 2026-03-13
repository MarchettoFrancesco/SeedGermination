
# Seed Germination CNN — Convnext_tiny Model (Balanced)

## Run
- Timestamp: 2025-10-11 12:50:49
- Epochs: 12
- Batch size: 32
- Target size: (224, 224)
- Architecture: convnext_tiny (pre-trained)
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
- Accuracy: 0.9541
- AUC: 0.9505

### Classification Report (0.5)
                precision    recall  f1-score   support

Not Germinated       0.98      0.96      0.97      2676
    Germinated       0.72      0.86      0.78       288

      accuracy                           0.95      2964
     macro avg       0.85      0.91      0.88      2964
  weighted avg       0.96      0.95      0.96      2964


### Confusion Matrix (0.5)
[[2580   96]
 [  40  248]]

## Threshold Tuning
- Chosen threshold: 0.465
- Validation F1 at chosen threshold: 0.725

### Classification Report (tuned)
                precision    recall  f1-score   support

Not Germinated       0.99      0.96      0.97      2676
    Germinated       0.71      0.86      0.78       288

      accuracy                           0.95      2964
     macro avg       0.85      0.91      0.88      2964
  weighted avg       0.96      0.95      0.96      2964


### Confusion Matrix (tuned)
[[2576  100]
 [  39  249]]
