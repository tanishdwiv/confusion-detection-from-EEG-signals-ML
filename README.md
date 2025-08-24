# EEG Confusion Detection with Transformer Models

This project implements **Transformer-based deep learning models** for detecting confusion from EEG signals.  
It leverages self-attention to capture long-term temporal dependencies and dynamically highlight important EEG features.  

---

## 🚀 Features
- Transformer, Conv-Transformer, and BiLSTM models.
- Leave-One-Subject-Out (LOSO) cross-validation.
- Early stopping and focal loss support.
- Per-subject Accuracy & F1 reporting.
- **Global ROC-AUC** (computed across all subjects).
- Visualization of subject-wise and average learning curves.

---

## 📂 Project Structure

---

## 🛠️ Setup (Google Colab Recommended)

1. Upload this project folder to your Colab session.
2. Install dependencies:
   ```bash
   pip install torch pandas scikit-learn matplotlib
python eeg_project/train.py --csv EEG_data.csv --model transformer
python eeg_project/plot_results.py
# Subject-wise curves
subject_id = 0
hist = pd.read_csv(f"{latest_run}/subject_{subject_id}/history.csv")
plt.plot(hist["epoch"], hist["val_acc"], label="Val Acc")
plt.plot(hist["epoch"], hist["val_f1"], label="Val F1")

# Average across subjects
# (See Step 6C in notebook)

