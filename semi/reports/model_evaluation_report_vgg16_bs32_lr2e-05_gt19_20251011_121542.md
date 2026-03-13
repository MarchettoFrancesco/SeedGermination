
# Seed Germination CNN — Vgg16 Model (Balanced)

## Run
- Timestamp: 2025-10-11 12:20:29
- Epochs: 12
- Batch size: 32
- Target size: (224, 224)
- Architecture: vgg16 (pre-trained)
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
- Accuracy: 0.9504
- AUC: 0.9735

### Classification Report (0.5)
                precision    recall  f1-score   support

Not Germinated       0.98      0.96      0.97      2676
    Germinated       0.70      0.86      0.77       288

      accuracy                           0.95      2964
     macro avg       0.84      0.91      0.87      2964
  weighted avg       0.96      0.95      0.95      2964


### Confusion Matrix (0.5)
[[2569  107]
 [  40  248]]

## Threshold Tuning
- Chosen threshold: 0.913
- Validation F1 at chosen threshold: 0.749

### Classification Report (tuned)
                precision    recall  f1-score   support

Not Germinated       0.97      0.98      0.98      2676
    Germinated       0.83      0.69      0.75       288

      accuracy                           0.96      2964
     macro avg       0.90      0.84      0.86      2964
  weighted avg       0.95      0.96      0.95      2964


### Confusion Matrix (tuned)
[[2634   42]
 [  89  199]]
