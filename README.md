# 🧠 Breast Cancer Detection using SVM
### Computer Aided Diagnosis from Mammogram Images

An end-to-end machine learning system for classifying **benign vs malignant**
breast tumors using Support Vector Machines (SVM), advanced preprocessing,
feature engineering, PCA dimensionality reduction, and ensemble strategies.

This repository demonstrates real-world ML engineering practices with focus on:

✔ reproducibility  
✔ modular design  
✔ automation  
✔ usability  
✔ clean project structure  

---

## 🎯 Motivation

Early detection of breast cancer significantly improves treatment success.
While radiologists are experts, AI-assisted systems can provide additional
decision support, reduce fatigue errors, and improve diagnostic consistency.

This project shows how carefully engineered classical ML pipelines can
still achieve strong performance in medical imaging tasks.

---

## 🚀 Project Highlights

- 📂 Automatic dataset organization  
- 🖼️ Radiology-focused image enhancement  
- 🧮 Multi-type feature extraction  
- 📉 PCA for dimensionality reduction  
- 🎯 Hyperparameter tuning  
- 📊 Confusion matrix & ROC evaluation  
- 💾 Saved models for reuse  
- 🖥️ Command-line predictions  
- 🧪 Environment verification tool  

---

## 🧠 System Pipeline

```
Raw Mammogram
→ Preprocessing
→ Feature Extraction
→ Scaling
→ PCA
→ SVM / Ensemble
→ Evaluation
→ Saved Model
→ Prediction
```

---

## 🏗️ Repository Structure

```
.
├── breast_cancer_svm.py
├── dataset_organizer.py
├── organize_dataset.py
├── train_model.py
├── predict.py
├── test_setup.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Model Performance

Results depend on dataset size and quality.

Typical experimental outcomes:

- 🎯 Accuracy: **90%+**
- Balanced precision & recall
- Stable cross-validation
- Clear ROC separation

---

## 🔬 Techniques Used

### Image Processing
- CLAHE contrast enhancement  
- Morphological filtering  
- Noise reduction  
- Histogram-based descriptors  

### Machine Learning
- Support Vector Machines (RBF / Linear)
- Ensemble methods
- Stratified train-test splits
- Robust scaling

### Dimensionality Reduction
- PCA with variance retention

---

## ⚙️ Installation

```bash
git clone https://github.com/UtsavOpal/Breast-Cancer-Detection-Using-SVM-and-Radiological-Data.git
cd Breast-Cancer-Detection-Using-SVM-and-Radiological-Data
pip install -r requirements.txt
```

---

## 🧪 Verify Setup

```bash
python test_setup.py
```

---

## 📂 Dataset Preparation

1. Download CBIS-DDSM or MIAS dataset from Kaggle.
2. Place extracted files into:

```
raw_dataset/
```

3. Run:

```bash
python organize_dataset.py
```

This creates:

```
organized_dataset/
├── benign/
└── malignant/
```

---

## 🏋️ Train the Model

```bash
python train_model.py
```

Training includes:

✔ scaling  
✔ PCA  
✔ hyperparameter search  
✔ cross-validation  
✔ evaluation  

Outputs are saved to:

```
models/
results/
```

---

## 🔮 Run Predictions

### Single image
```bash
python predict.py --image path/to/image.png
```

### Folder
```bash
python predict.py --folder path/to/folder
```

---

## 💾 Model Reuse

Once trained, the saved model can be used for future predictions
without retraining.

---

## 🧩 Engineering Principles Demonstrated

- Modular ML architecture  
- Reproducible pipelines  
- Data/code separation  
- Automation scripts  
- CLI interfaces  
- Scalable for future deep learning integration  

---

## 📈 Challenges Solved

- Extremely high-dimensional inputs  
- Dataset imbalance  
- Medical image noise  
- Risk of overfitting  

Addressed using:

✔ PCA  
✔ augmentation  
✔ ensemble modeling  
✔ robust preprocessing  

---

## 🔮 Future Improvements

- CNN / deep learning comparison  
- Explainability modules  
- Clinical user interface  
- REST API service  
- Cloud deployment  
- Multi-class classification  

---

## 🎓 Learning Outcomes

This project strengthened my understanding of:

- ML system design  
- medical imaging workflows  
- feature engineering  
- evaluation strategies  
- production-style repositories  

---

## 👨‍💻 Author

**Utsav Opal**  
SRM University

---

If you found this interesting, feel free to star ⭐ the repository.
