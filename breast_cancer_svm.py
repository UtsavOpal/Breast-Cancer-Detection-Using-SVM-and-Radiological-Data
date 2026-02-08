"""
BREAST CANCER DETECTION USING SVM - COMPLETE MODEL
===================================================
Complete implementation of breast cancer detection system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, roc_auc_score)
from sklearn.decomposition import PCA
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try importing DICOM support
try:
    import pydicom
    DICOM_SUPPORT = True
except ImportError:
    DICOM_SUPPORT = False


class BreastCancerSVMDetector:
    """
    Breast Cancer Detection System using SVM and Radiological Images
    """
    
    def __init__(self, img_size=(128, 128)):
        """
        Initialize the detector
        
        Parameters:
        -----------
        img_size : tuple
            Target size for images (height, width)
        """
        self.img_size = img_size
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.class_names = ['Benign', 'Malignant']
        
        print(f"✅ Detector initialized with image size: {img_size}")
    
    def load_image(self, img_path):
        """Load image from various formats"""
        img_path = str(img_path)
        
        # Try DICOM format
        if img_path.lower().endswith('.dcm') and DICOM_SUPPORT:
            try:
                dicom = pydicom.dcmread(img_path)
                img = dicom.pixel_array.astype(float)
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                return img
            except:
                pass
        
        # Try standard formats
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img
    
    def preprocess_image(self, img):
        """Preprocess a single image"""
        if img is None:
            return None
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Histogram equalization
        img = cv2.equalizeHist(img)
        
        # Gaussian blur
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Flatten
        return img.flatten()
    
    def load_dataset_from_folder(self, benign_folder, malignant_folder):
        """
        Load images from benign and malignant folders
        
        Parameters:
        -----------
        benign_folder : str
            Path to folder with benign images
        malignant_folder : str
            Path to folder with malignant images
        
        Returns:
        --------
        X : numpy array
            Feature matrix
        y : numpy array
            Labels (0=Benign, 1=Malignant)
        filenames : list
            List of filenames
        """
        features = []
        labels = []
        filenames = []
        
        print("\n" + "="*70)
        print("LOADING DATASET")
        print("="*70)
        
        # Load benign images
        benign_path = Path(benign_folder)
        if not benign_path.exists():
            raise ValueError(f"Benign folder not found: {benign_folder}")
        
        print(f"\nLoading BENIGN images from: {benign_folder}")
        benign_files = list(benign_path.glob('*'))
        benign_files = [f for f in benign_files if f.suffix.lower() in 
                       ['.png', '.jpg', '.jpeg', '.dcm', '.pgm', '.tif', '.tiff']]
        
        for img_file in tqdm(benign_files, desc="Benign"):
            img = self.load_image(img_file)
            img_features = self.preprocess_image(img)
            if img_features is not None:
                features.append(img_features)
                labels.append(0)  # 0 = Benign
                filenames.append(img_file.name)
        
        benign_count = len([l for l in labels if l == 0])
        print(f"✓ Loaded {benign_count} benign images")
        
        # Load malignant images
        malignant_path = Path(malignant_folder)
        if not malignant_path.exists():
            raise ValueError(f"Malignant folder not found: {malignant_folder}")
        
        print(f"\nLoading MALIGNANT images from: {malignant_folder}")
        malignant_files = list(malignant_path.glob('*'))
        malignant_files = [f for f in malignant_files if f.suffix.lower() in 
                          ['.png', '.jpg', '.jpeg', '.dcm', '.pgm', '.tif', '.tiff']]
        
        for img_file in tqdm(malignant_files, desc="Malignant"):
            img = self.load_image(img_file)
            img_features = self.preprocess_image(img)
            if img_features is not None:
                features.append(img_features)
                labels.append(1)  # 1 = Malignant
                filenames.append(img_file.name)
        
        malignant_count = len([l for l in labels if l == 1])
        print(f"✓ Loaded {malignant_count} malignant images")
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"\n" + "="*70)
        print(f"DATASET SUMMARY")
        print(f"="*70)
        print(f"Total images: {len(X)}")
        print(f"Benign (0): {benign_count} ({benign_count/len(X)*100:.1f}%)")
        print(f"Malignant (1): {malignant_count} ({malignant_count/len(X)*100:.1f}%)")
        print(f"Feature dimensions: {X.shape[1]}")
        
        return X, y, filenames
    
    def apply_pca(self, X, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        print("\n" + "="*70)
        print("FEATURE EXTRACTION (PCA)")
        print("="*70)
        print(f"Original dimensions: {X.shape[1]}")
        
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        print(f"Reduced dimensions: {X_pca.shape[1]}")
        print(f"Explained variance: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        return X_pca
    
    def train(self, X, y, use_grid_search=True, test_size=0.2):
        """
        Train the SVM model
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Labels
        use_grid_search : bool
            Whether to perform hyperparameter tuning
        test_size : float
            Proportion for testing
        
        Returns:
        --------
        results : dict
            Training results and metrics
        """
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nDataset Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Training - Benign: {np.sum(y_train==0)}, Malignant: {np.sum(y_train==1)}")
        print(f"  Testing  - Benign: {np.sum(y_test==0)}, Malignant: {np.sum(y_test==1)}")
        
        # Scale features
        print("\nApplying feature scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA
        X_train_pca = self.apply_pca(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Train SVM
        if use_grid_search:
            print("\n" + "="*70)
            print("HYPERPARAMETER TUNING")
            print("="*70)
            
            param_grid = {
                'C': [1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
            }
            
            svm = SVC(random_state=42, probability=True)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            
            print("Starting grid search (this may take a few minutes)...")
            grid_search.fit(X_train_pca, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"\n✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV score: {grid_search.best_score_*100:.2f}%")
        else:
            print("\nTraining SVM...")
            self.model = SVC(kernel='rbf', C=10, gamma='scale',
                           random_state=42, probability=True)
            self.model.fit(X_train_pca, y_train)
            print("✅ Training complete")
        
        # Cross-validation
        print("\n" + "="*70)
        print("CROSS-VALIDATION (5-Fold)")
        print("="*70)
        cv_scores = cross_val_score(self.model, X_train_pca, y_train,
                                    cv=5, scoring='accuracy')
        print(f"CV Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
        print(f"Mean: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        
        # Evaluate
        metrics = self.evaluate(X_test_pca, y_test)
        
        return {
            'X_train': X_train_pca,
            'X_test': X_test_pca,
            'y_train': y_train,
            'y_test': y_test,
            'metrics': metrics,
            'cv_scores': cv_scores
        }
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Display
        print("\n📊 PERFORMANCE METRICS")
        print("="*70)
        print(f"  Accuracy:  {accuracy*100:6.2f}%  {'✅ PASSED' if accuracy >= 0.85 else '⚠️  BELOW TARGET'}")
        print(f"  Precision: {precision*100:6.2f}%")
        print(f"  Recall:    {recall*100:6.2f}%")
        print(f"  F1-Score:  {f1*100:6.2f}%")
        print(f"  ROC-AUC:   {roc_auc:6.4f}")
        print("="*70)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📋 CONFUSION MATRIX")
        print("="*70)
        print(f"                  Predicted")
        print(f"                Benign  Malignant")
        print(f"Actual Benign      {cm[0][0]:4d}     {cm[0][1]:4d}")
        print(f"       Malignant   {cm[1][0]:4d}     {cm[1][1]:4d}")
        
        # Classification Report
        print("\n📈 CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_test, y_pred,
                                   target_names=['Benign (0)', 'Malignant (1)'],
                                   digits=4))
        
        # Plot visualizations
        self.plot_confusion_matrix(cm)
        self.plot_roc_curve(y_test, y_pred_proba, roc_auc)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign (0)', 'Malignant (1)'],
                   yticklabels=['Benign (0)', 'Malignant (1)'],
                   annot_kws={'size': 16, 'weight': 'bold'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save to results folder
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✅ Confusion matrix saved: results/confusion_matrix.png")
        plt.close()
    
    def plot_roc_curve(self, y_test, y_pred_proba, roc_auc):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2.5,
                label=f'SVM (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random (AUC = 0.5000)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
        print("✅ ROC curve saved: results/roc_curve.png")
        plt.close()
    
    def predict(self, image_path, verbose=True):
        """
        Predict single image
        
        Returns:
        --------
        result : dict
            Prediction results with labels and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        try:
            # Load and preprocess
            img = self.load_image(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            img_features = self.preprocess_image(img)
            img_features = img_features.reshape(1, -1)
            
            # Scale and PCA
            img_scaled = self.scaler.transform(img_features)
            img_pca = self.pca.transform(img_scaled)
            
            # Predict
            prediction = self.model.predict(img_pca)[0]
            proba = self.model.predict_proba(img_pca)[0]
            
            result = {
                'class_label': int(prediction),
                'class_name': self.class_names[prediction],
                'confidence': float(proba[prediction] * 100),
                'probabilities': {
                    'Benign': float(proba[0] * 100),
                    'Malignant': float(proba[1] * 100)
                }
            }
            
            if verbose:
                print("\n" + "="*70)
                print("PREDICTION RESULT")
                print("="*70)
                print(f"Image: {Path(image_path).name}")
                print(f"Prediction: {result['class_name']} (Label: {result['class_label']})")
                print(f"Confidence: {result['confidence']:.2f}%")
                print(f"\nProbabilities:")
                print(f"  Benign (0):    {result['probabilities']['Benign']:.2f}%")
                print(f"  Malignant (1): {result['probabilities']['Malignant']:.2f}%")
                print("="*70)
            
            return result
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            return None
    
    def batch_predict(self, image_paths):
        """Predict multiple images"""
        results = []
        print(f"\nPredicting {len(image_paths)} images...")
        
        for i, img_path in enumerate(image_paths):
            result = self.predict(img_path, verbose=False)
            if result:
                results.append(result)
                print(f"  [{i+1}/{len(image_paths)}] {Path(img_path).name}: "
                      f"{result['class_name']} ({result['confidence']:.1f}%)")
        
        return results


# Test if module is working
if __name__ == "__main__":
    print("\n" + "="*70)
    print("BREAST CANCER SVM DETECTOR - MODULE TEST")
    print("="*70)
    print("\n✅ Module imported successfully!")
    print("✅ BreastCancerSVMDetector class is available")
    print("\nYou can now run:")
    print("  python train_model.py")