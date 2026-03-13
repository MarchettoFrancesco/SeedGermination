
# Seed Germination CNN — Resnet50 Model (Balanced)

## Run
- Timestamp: 2025-10-11 12:14:53
- Epochs: 12
- Batch size: 32
- Target size: (224, 224)
- Architecture: resnet50 (pre-trained)
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
- Accuracy: 0.9464
- AUC: 0.9760

### Classification Report (0.5)
                precision    recall  f1-score   support

Not Germinated       0.98      0.96      0.97      2676
    Germinated       0.69      0.83      0.75       288

      accuracy                           0.95      2964
     macro avg       0.83      0.89      0.86      2964
  weighted avg       0.95      0.95      0.95      2964


### Confusion Matrix (0.5)
[[2567  109]
 [  50  238]]

## Threshold Tuning
- Chosen threshold: 0.858
- Validation F1 at chosen threshold: 0.689

### Classification Report (tuned)
                precision    recall  f1-score   support

Not Germinated       0.98      0.97      0.98      2676
    Germinated       0.77      0.78      0.77       288

      accuracy                           0.96      2964
     macro avg       0.87      0.88      0.87      2964
  weighted avg       0.96      0.96      0.96      2964


### Confusion Matrix (tuned)
[[2608   68]
 [  64  224]]
