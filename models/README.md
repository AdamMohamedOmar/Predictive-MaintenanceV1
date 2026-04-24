# Models directory

Trained model artifacts live here. **Gitignored** — regenerate with training scripts.

Expected files as project progresses:

| File | Produced by | Week |
|---|---|---|
| `rf_classifier_v1.pkl` | `scripts/train_classifier.py` | 3 |
| `xgb_classifier_v1.pkl` | `scripts/train_classifier.py` | 4 |
| `xgb_classifier_v1.onnx` | `scripts/export_onnx.py` | 6 |
| `rf_forecaster_<class>.pkl` | `scripts/train_forecaster.py` | 5 |
| `cnn_forecaster_<class>.pt` | `scripts/train_forecaster.py` | 6 |
| `cnn_forecaster_<class>.onnx` | `scripts/export_onnx.py` | 6 |

To regenerate everything from scratch after a fresh clone:

```bash
python scripts/generate_dataset.py
python scripts/train_classifier.py
python scripts/train_forecaster.py
python scripts/export_onnx.py
```
