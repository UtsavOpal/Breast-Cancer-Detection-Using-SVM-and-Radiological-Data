# """
# MODEL TRAINING SCRIPT
# =====================
# Train the breast cancer detection SVM model.

# Usage:
#     python train_model.py
# """

# from breast_cancer_svm import BreastCancerSVMDetector
# import pickle
# import os
# import sys
# from datetime import datetime

# def check_dataset():
#     """Check if organized dataset exists"""
#     benign_path = 'organized_dataset/benign'
#     malignant_path = 'organized_dataset/malignant'
    
#     if not os.path.exists('organized_dataset'):
#         print("❌ Error: 'organized_dataset' folder not found!")
#         print("\n📝 Please run dataset organization first:")
#         print("   python organize_dataset.py")
#         return False
    
#     if not os.path.exists(benign_path) or not os.path.exists(malignant_path):
#         print("❌ Error: benign/ or malignant/ folders not found!")
#         print("\n📝 Please run dataset organization first:")
#         print("   python organize_dataset.py")
#         return False
    
#     benign_count = len([f for f in os.listdir(benign_path) 
#                        if os.path.isfile(os.path.join(benign_path, f))])
#     malignant_count = len([f for f in os.listdir(malignant_path) 
#                           if os.path.isfile(os.path.join(malignant_path, f))])
    
#     if benign_count == 0 or malignant_count == 0:
#         print("❌ Error: Dataset folders are empty!")
#         print(f"   Benign images: {benign_count}")
#         print(f"   Malignant images: {malignant_count}")
#         return False
    
#     print(f"✅ Dataset found:")
#     print(f"   Benign images: {benign_count}")
#     print(f"   Malignant images: {malignant_count}")
#     print(f"   Total images: {benign_count + malignant_count}")
    
#     return True

# def main():
#     print("\n" + "="*70)
#     print("BREAST CANCER DETECTION - MODEL TRAINING")
#     print("="*70)
    
#     # Check dataset
#     print("\n1. Checking dataset...")
#     if not check_dataset():
#         sys.exit(1)
    
#     # Create directories
#     print("\n2. Creating output directories...")
#     os.makedirs('results', exist_ok=True)
#     os.makedirs('models', exist_ok=True)
#     print("   ✅ Created: results/")
#     print("   ✅ Created: models/")
    
#     # Initialize detector
#     print("\n3. Initializing detector...")
#     print("   Image size: 128x128 pixels")
#     detector = BreastCancerSVMDetector(img_size=(128, 128))
#     print("   ✅ Detector initialized")
    
#     # Load dataset
#     print("\n4. Loading dataset...")
#     print("   This may take a few minutes for large datasets...")
#     try:
#         X, y, filenames = detector.load_dataset_from_folder(
#             benign_folder='organized_dataset/benign',
#             malignant_folder='organized_dataset/malignant'
#         )
#         print(f"\n   ✅ Loaded {len(X)} images successfully!")
#     except Exception as e:
#         print(f"\n   ❌ Error loading dataset: {str(e)}")
#         sys.exit(1)
    
#     # Check dataset size
#     if len(X) < 50:
#         print("\n   ⚠️  Warning: Dataset is very small (<50 images)")
#         print("   Model accuracy may be low. Consider adding more images.")
#         response = input("\n   Continue anyway? (y/n): ").lower()
#         if response != 'y':
#             print("   Training cancelled.")
#             sys.exit(0)
    
#     # Train model
#     print("\n5. Training model...")
#     print("   This will take 5-15 minutes depending on dataset size")
#     print("   - Performing data split (80% train, 20% test)")
#     print("   - Applying feature scaling")
#     print("   - Running PCA for dimensionality reduction")
#     print("   - Hyperparameter tuning with Grid Search")
#     print("   - 5-fold cross-validation")
#     print("\n" + "-"*70)
    
#     try:
#         results = detector.train(X, y, use_grid_search=True, test_size=0.2)
#     except Exception as e:
#         print(f"\n❌ Error during training: {str(e)}")
#         sys.exit(1)
    
#     # Save model
#     print("\n6. Saving model...")
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_path = f'models/svm_model_{timestamp}.pkl'
#     latest_model_path = 'models/svm_model_latest.pkl'
    
#     try:
#         model_data = {
#             'model': detector.model,
#             'scaler': detector.scaler,
#             'pca': detector.pca,
#             'img_size': detector.img_size,
#             'class_names': detector.class_names,
#             'training_date': timestamp,
#             'metrics': results['metrics']
#         }
        
#         # Save with timestamp
#         with open(model_path, 'wb') as f:
#             pickle.dump(model_data, f)
#         print(f"   ✅ Model saved: {model_path}")
        
#         # Save as latest
#         with open(latest_model_path, 'wb') as f:
#             pickle.dump(model_data, f)
#         print(f"   ✅ Latest model: {latest_model_path}")
        
#     except Exception as e:
#         print(f"   ❌ Error saving model: {str(e)}")
    
#     # Summary
#     print("\n" + "="*70)
#     print("TRAINING COMPLETE!")
#     print("="*70)
    
#     metrics = results['metrics']
#     print(f"\n📊 Final Model Performance:")
#     print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
#     print(f"   Precision: {metrics['precision']*100:.2f}%")
#     print(f"   Recall:    {metrics['recall']*100:.2f}%")
#     print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
#     print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    
#     if metrics['accuracy'] >= 0.85:
#         print("\n   ✅ Model MEETS the 85% accuracy requirement!")
#     else:
#         print(f"\n   ⚠️  Model accuracy ({metrics['accuracy']*100:.2f}%) is below 85%")
#         print("   Consider: Adding more data or adjusting parameters")
    
#     print(f"\n📁 Generated Files:")
#     print(f"   • {model_path}")
#     print(f"   • {latest_model_path}")
#     print(f"   • results/confusion_matrix.png")
#     print(f"   • results/roc_curve.png")
#     print(f"   • results/prediction_distribution.png")
    
#     print(f"\n📝 Next Steps:")
#     print(f"   1. Check 'results/' folder for visualizations")
#     print(f"   2. Run 'python predict.py' to make predictions")
#     print(f"   3. Test model on new images")
    
#     print("\n" + "="*70)

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n\n⚠️  Training interrupted by user")
#         sys.exit(0)
#     except Exception as e:
#         print(f"\n\n❌ Unexpected error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

"""
ADVANCED TRAINING FOR 90%+ ACCURACY
====================================
Advanced techniques:
- Data augmentation
- Ensemble methods
- Feature engineering
- Advanced SVM with RBF kernel
- Stratified sampling
"""

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AdvancedImagePreprocessor:
    """Advanced image preprocessing for maximum accuracy"""
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
    
    def preprocess(self, img):
        """Apply advanced preprocessing pipeline"""
        if img is None:
            return None
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # 1. CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # 2. Morphological operations to enhance structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # 3. Bilateral filter (preserve edges, reduce noise)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 4. Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def augment_image(self, img):
        """Create augmented versions of image"""
        augmented = [img]  # Original
        
        # Horizontal flip
        augmented.append(cv2.flip(img, 1))
        
        # Small rotation
        center = (img.shape[1] // 2, img.shape[0] // 2)
        for angle in [5, -5]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            augmented.append(rotated)
        
        # Slight zoom
        zoomed = cv2.resize(img, None, fx=1.1, fy=1.1)
        h, w = img.shape[:2]
        zh, zw = zoomed.shape[:2]
        x = (zw - w) // 2
        y = (zh - h) // 2
        zoomed = zoomed[y:y+h, x:x+w]
        augmented.append(zoomed)
        
        return augmented
    
    def extract_features(self, img):
        """Extract comprehensive features from image"""
        features = []
        
        # 1. Raw pixel features
        features.extend(img.flatten())
        
        # 2. Edge features (Sobel)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([
            edge_magnitude.mean(),
            edge_magnitude.std(),
            edge_magnitude.max()
        ])
        
        # 3. Texture features (Local Binary Pattern approximation)
        hist = cv2.calcHist([img], [0], None, [32], [0, 1])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # 4. Statistical features by regions
        h, w = img.shape
        regions = [
            img[0:h//2, 0:w//2],      # Top-left
            img[0:h//2, w//2:w],      # Top-right
            img[h//2:h, 0:w//2],      # Bottom-left
            img[h//2:h, w//2:w]       # Bottom-right
        ]
        for region in regions:
            features.extend([
                region.mean(),
                region.std(),
                region.max(),
                region.min()
            ])
        
        return np.array(features)

def load_dataset_advanced(benign_folder, malignant_folder, img_size=(224, 224), augment=False):
    """Load dataset with advanced preprocessing"""
    
    preprocessor = AdvancedImagePreprocessor(img_size)
    
    features_list = []
    labels_list = []
    
    print("\n" + "="*70)
    print("LOADING DATASET WITH ADVANCED PREPROCESSING")
    print("="*70)
    
    # Load benign images
    benign_path = Path(benign_folder)
    benign_files = [f for f in benign_path.glob('*') 
                   if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.dcm', '.pgm', '.tif', '.tiff']]
    
    print(f"\nProcessing BENIGN images...")
    for img_file in tqdm(benign_files, desc="Benign"):
        try:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img = preprocessor.preprocess(img)
            if img is None:
                continue
            
            if augment:
                # Add augmented versions (increases training data)
                augmented_imgs = preprocessor.augment_image(img)
                for aug_img in augmented_imgs:
                    features = preprocessor.extract_features(aug_img)
                    features_list.append(features)
                    labels_list.append(0)
            else:
                features = preprocessor.extract_features(img)
                features_list.append(features)
                labels_list.append(0)
                
        except Exception as e:
            continue
    
    benign_count = sum(1 for l in labels_list if l == 0)
    print(f"✓ Loaded {benign_count} benign samples")
    
    # Load malignant images
    malignant_path = Path(malignant_folder)
    malignant_files = [f for f in malignant_path.glob('*')
                      if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.dcm', '.pgm', '.tif', '.tiff']]
    
    print(f"\nProcessing MALIGNANT images...")
    for img_file in tqdm(malignant_files, desc="Malignant"):
        try:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img = preprocessor.preprocess(img)
            if img is None:
                continue
            
            if augment:
                augmented_imgs = preprocessor.augment_image(img)
                for aug_img in augmented_imgs:
                    features = preprocessor.extract_features(aug_img)
                    features_list.append(features)
                    labels_list.append(1)
            else:
                features = preprocessor.extract_features(img)
                features_list.append(features)
                labels_list.append(1)
                
        except Exception as e:
            continue
    
    malignant_count = sum(1 for l in labels_list if l == 1)
    print(f"✓ Loaded {malignant_count} malignant samples")
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total samples: {len(X)}")
    print(f"Benign (0): {benign_count} ({benign_count/len(X)*100:.1f}%)")
    print(f"Malignant (1): {malignant_count} ({malignant_count/len(X)*100:.1f}%)")
    print(f"Feature dimensions: {X.shape[1]}")
    
    return X, y

def balance_classes(X, y):
    """Balance classes using SMOTE-like oversampling"""
    from collections import Counter
    
    counts = Counter(y)
    print(f"\nOriginal class distribution: {dict(counts)}")
    
    # Find minority class
    minority_class = 1 if counts[1] < counts[0] else 0
    majority_class = 0 if minority_class == 1 else 1
    
    minority_count = counts[minority_class]
    majority_count = counts[majority_class]
    
    if majority_count / minority_count > 1.5:
        print("⚡ Balancing classes...")
        
        # Get indices
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Oversample minority class with noise
        samples_needed = majority_count - minority_count
        oversample_indices = np.random.choice(minority_indices, samples_needed, replace=True)
        
        X_minority = X[oversample_indices]
        # Add small random noise to oversampled data
        noise = np.random.normal(0, 0.01, X_minority.shape)
        X_minority = X_minority + noise
        
        X_balanced = np.vstack([X, X_minority])
        y_balanced = np.hstack([y, np.full(samples_needed, minority_class)])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        print(f"Balanced distribution: {dict(Counter(y_balanced))}")
        return X_balanced, y_balanced
    
    return X, y

def train_advanced_ensemble(X, y):
    """Train advanced ensemble model for maximum accuracy"""
    
    print("\n" + "="*70)
    print("ADVANCED ENSEMBLE TRAINING")
    print("="*70)
    
    # Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.12, random_state=42, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing:  {len(X_test)} samples")
    
    # Use RobustScaler (better for outliers)
    print("\n⚡ Applying robust scaling...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA with optimal components
    print("\n⚡ Applying PCA...")
    pca = PCA(n_components=0.99, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"   Dimensions: {X_train_scaled.shape[1]} → {X_train_pca.shape[1]}")
    print(f"   Variance retained: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Train multiple SVM models with different parameters
    print("\n⚡ Training ensemble of SVM models...")
    
    models = []
    
    # Model 1: High C, scale gamma
    print("   Training Model 1: High regularization...")
    svm1 = SVC(C=100, kernel='rbf', gamma='scale', probability=True, 
               random_state=42, cache_size=2000, class_weight='balanced')
    svm1.fit(X_train_pca, y_train)
    acc1 = svm1.score(X_test_pca, y_test)
    print(f"      Accuracy: {acc1*100:.2f}%")
    models.append(('svm1', svm1))
    
    # Model 2: Very high C, auto gamma
    print("   Training Model 2: Very high C...")
    svm2 = SVC(C=500, kernel='rbf', gamma='auto', probability=True,
               random_state=43, cache_size=2000, class_weight='balanced')
    svm2.fit(X_train_pca, y_train)
    acc2 = svm2.score(X_test_pca, y_test)
    print(f"      Accuracy: {acc2*100:.2f}%")
    models.append(('svm2', svm2))
    
    # Model 3: Moderate C, specific gamma
    print("   Training Model 3: Optimized gamma...")
    svm3 = SVC(C=200, kernel='rbf', gamma=0.01, probability=True,
               random_state=44, cache_size=2000, class_weight='balanced')
    svm3.fit(X_train_pca, y_train)
    acc3 = svm3.score(X_test_pca, y_test)
    print(f"      Accuracy: {acc3*100:.2f}%")
    models.append(('svm3', svm3))
    
    # Create voting ensemble
    print("\n⚡ Creating voting ensemble...")
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',  # Use probability voting
        weights=[1, 1, 1]  # Equal weights
    )
    ensemble.fit(X_train_pca, y_train)
    
    # Evaluate ensemble
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION")
    print("="*70)
    
    y_pred = ensemble.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 INDIVIDUAL MODEL ACCURACIES:")
    print(f"   Model 1: {acc1*100:.2f}%")
    print(f"   Model 2: {acc2*100:.2f}%")
    print(f"   Model 3: {acc3*100:.2f}%")
    print(f"\n🎯 ENSEMBLE ACCURACY: {accuracy*100:.2f}%")
    
    if accuracy >= 0.90:
        print(f"\n   ✅✅✅ SUCCESS! Achieved {accuracy*100:.2f}% accuracy!")
    elif accuracy >= 0.85:
        print(f"\n   ✅✅ Good! {accuracy*100:.2f}% accuracy")
    else:
        print(f"\n   ⚠️  {accuracy*100:.2f}% - trying to improve...")
    
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=['Benign (0)', 'Malignant (1)'],
                                       digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\n📋 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                Benign  Malignant")
    print(f"Actual Benign      {cm[0][0]:4d}     {cm[0][1]:4d}")
    print(f"       Malignant   {cm[1][0]:4d}     {cm[1][1]:4d}")
    
    # Calculate per-class accuracy
    benign_accuracy = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    malignant_accuracy = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    
    print(f"\n📈 Per-Class Accuracy:")
    print(f"   Benign:    {benign_accuracy*100:.2f}%")
    print(f"   Malignant: {malignant_accuracy*100:.2f}%")
    
    return ensemble, scaler, pca, accuracy

def main():
    print("\n" + "="*70)
    print("ADVANCED TRAINING - TARGET: 90%+ ACCURACY")
    print("Using: Ensemble Methods + Data Augmentation + Feature Engineering")
    print("="*70)
    
    # Check dataset
    benign_path = 'organized_dataset/benign'
    malignant_path = 'organized_dataset/malignant'
    
    if not os.path.exists(benign_path) or not os.path.exists(malignant_path):
        print("❌ Error: Dataset not found!")
        sys.exit(1)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Ask about data augmentation
    print("\n" + "="*70)
    print("DATA AUGMENTATION OPTIONS")
    print("="*70)
    print("Data augmentation creates additional training samples by:")
    print("  - Flipping images")
    print("  - Rotating slightly")
    print("  - Zooming")
    print("\nThis increases training data 5x but takes longer to process.")
    print("Recommended for datasets < 1000 images per class.")
    
    augment = input("\nUse data augmentation? (y/n) [y]: ").lower().strip()
    augment = augment != 'n'
    
    # Load dataset
    print("\n" + "="*70)
    print("PHASE 1: LOADING DATA")
    print("="*70)
    
    X, y = load_dataset_advanced(
        benign_path,
        malignant_path,
        img_size=(224, 224),
        augment=augment
    )
    
    # Balance classes
    print("\n" + "="*70)
    print("PHASE 2: BALANCING CLASSES")
    print("="*70)
    X, y = balance_classes(X, y)
    
    # Train model
    print("\n" + "="*70)
    print("PHASE 3: TRAINING ENSEMBLE MODEL")
    print("="*70)
    
    model, scaler, pca, accuracy = train_advanced_ensemble(X, y)
    
    # Save model
    print("\n" + "="*70)
    print("PHASE 4: SAVING MODEL")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/svm_model_advanced_{timestamp}.pkl'
    latest_path = 'models/svm_model_latest.pkl'
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'img_size': (224, 224),
        'class_names': ['Benign', 'Malignant'],
        'training_date': timestamp,
        'metrics': {'accuracy': accuracy},
        'version': 'advanced_ensemble'
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    with open(latest_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Saved: {model_path}")
    print(f"✅ Saved: {latest_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n🎯 FINAL ACCURACY: {accuracy*100:.2f}%")
    
    if accuracy >= 0.90:
        print("\n🎉🎉🎉 SUCCESS! Target achieved!")
    elif accuracy >= 0.88:
        print("\n✅ Very close! Try with data augmentation if not used.")
    elif accuracy >= 0.85:
        print("\n💡 Tips to reach 90%:")
        print("   1. Run again with data augmentation (y)")
        print("   2. Add more diverse images")
        print("   3. Check for mislabeled images")
    
    print(f"\n📝 Next: python predict.py --test")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)