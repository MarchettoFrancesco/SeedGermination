
# Seed Germination CNN — Efficientnet_b0 Model (Balanced)

## Run
- Timestamp: 2025-10-11 12:34:57
- Epochs: 12
- Batch size: 32
- Target size: (224, 224)
- Architecture: efficientnet_b0 (pre-trained)
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
- Accuracy: 0.9588
- AUC: 0.9649

### Classification Report (0.5)
                precision    recall  f1-score   support

Not Germinated       0.99      0.97      0.98      2676
    Germinated       0.75      0.86      0.80       288

      accuracy                           0.96      2964
     macro avg       0.87      0.92      0.89      2964
  weighted avg       0.96      0.96      0.96      2964


### Confusion Matrix (0.5)
[[2593   83]
 [  39  249]]

## Threshold Tuning
- Chosen threshold: 0.738
- Validation F1 at chosen threshold: 0.721

### Classification Report (tuned)
                precision    recall  f1-score   support

Not Germinated       0.98      0.98      0.98      2676
    Germinated       0.82      0.81      0.81       288

      accuracy                           0.96      2964
     macro avg       0.90      0.89      0.90      2964
  weighted avg       0.96      0.96      0.96      2964


### Confusion Matrix (tuned)
[[2625   51]
 [  56  232]]
