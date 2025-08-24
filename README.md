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

<img width="640" height="480" alt="download" src="https://github.com/user-attachments/assets/93d02ad9-b73c-400e-a985-af258236457a" />


<img width="691" height="470" alt="download (1)" src="https://github.com/user-attachments/assets/47275aa6-f5eb-4a8e-9f2e-45f97cf50df8" />
