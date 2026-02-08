"""
ADVANCED PREDICTION SCRIPT
===========================
Compatible with the advanced trained model (93.27% accuracy)
"""

import numpy as np
import cv2
import pickle
import sys
import os
import glob
import argparse
from pathlib import Path

class AdvancedImagePreprocessor:
    """Same preprocessing as training"""
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
    
    def preprocess(self, img):
        """Apply advanced preprocessing pipeline"""
        if img is None:
            return None
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Bilateral filter
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def extract_features(self, img):
        """Extract comprehensive features"""
        features = []
        
        # 1. Raw pixel features
        features.extend(img.flatten())
        
        # 2. Edge features
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([
            edge_magnitude.mean(),
            edge_magnitude.std(),
            edge_magnitude.max()
        ])
        
        # 3. Texture features
        img_uint8 = (img * 255).astype(np.uint8)
        hist = cv2.calcHist([img_uint8], [0], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # 4. Regional statistics
        h, w = img.shape
        regions = [
            img[0:h//2, 0:w//2],
            img[0:h//2, w//2:w],
            img[h//2:h, 0:w//2],
            img[h//2:h, w//2:w]
        ]
        for region in regions:
            features.extend([
                region.mean(),
                region.std(),
                region.max(),
                region.min()
            ])
        
        return np.array(features)

def load_trained_model(model_path='models/svm_model_latest.pkl'):
    """Load the advanced trained model"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        
        if 'training_date' in saved_data:
            print(f"   Trained on: {saved_data['training_date']}")
        if 'metrics' in saved_data:
            metrics = saved_data['metrics']
            if 'accuracy' in metrics:
                print(f"   Model Accuracy: {metrics['accuracy']*100:.2f}%")
        
        return saved_data
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        sys.exit(1)

def predict_image(image_path, model_data, verbose=True):
    """Predict a single image"""
    
    try:
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get image size from model
        img_size = model_data.get('img_size', (224, 224))
        
        # Preprocess
        preprocessor = AdvancedImagePreprocessor(img_size)
        img_processed = preprocessor.preprocess(img)
        
        if img_processed is None:
            raise ValueError("Preprocessing failed")
        
        # Extract features
        features = preprocessor.extract_features(img_processed)
        features = features.reshape(1, -1)
        
        # Scale
        scaler = model_data['scaler']
        features_scaled = scaler.transform(features)
        
        # PCA
        pca = model_data['pca']
        features_pca = pca.transform(features_scaled)
        
        # Predict
        model = model_data['model']
        prediction = model.predict(features_pca)[0]
        proba = model.predict_proba(features_pca)[0]
        
        class_names = model_data.get('class_names', ['Benign', 'Malignant'])
        
        result = {
            'class_label': int(prediction),
            'class_name': class_names[prediction],
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
        if verbose:
            print(f"❌ Error predicting {Path(image_path).name}: {str(e)}")
        return None

def predict_folder(folder_path, model_data):
    """Predict all images in a folder"""
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return None
    
    # Find images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.dcm', '*.pgm', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Found {len(image_files)} images in {folder_path}")
    print('='*70)
    
    results = []
    for i, img_path in enumerate(image_files):
        result = predict_image(img_path, model_data, verbose=False)
        if result:
            results.append(result)
            print(f"  [{i+1}/{len(image_files)}] {Path(img_path).name}: "
                  f"{result['class_name']} ({result['confidence']:.1f}%)")
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print("PREDICTION SUMMARY")
        print('='*70)
        
        benign_count = sum(1 for r in results if r['class_label'] == 0)
        malignant_count = sum(1 for r in results if r['class_label'] == 1)
        
        print(f"Total predictions: {len(results)}")
        print(f"Benign (0):    {benign_count} ({benign_count/len(results)*100:.1f}%)")
        print(f"Malignant (1): {malignant_count} ({malignant_count/len(results)*100:.1f}%)")
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"Average confidence: {avg_confidence:.2f}%")
    
    return results

def test_on_validation_set(model_data):
    """Test model on samples from organized dataset"""
    
    print(f"\n{'='*70}")
    print("TESTING ON VALIDATION SAMPLES")
    print('='*70)
    
    # Get sample images
    benign_images = glob.glob('organized_dataset/benign/*')[:10]
    malignant_images = glob.glob('organized_dataset/malignant/*')[:10]
    
    if not benign_images and not malignant_images:
        print("❌ No validation images found in organized_dataset/")
        return
    
    test_images = benign_images + malignant_images
    print(f"Testing on {len(test_images)} validation samples...")
    print(f"   Benign: {len(benign_images)}")
    print(f"   Malignant: {len(malignant_images)}")
    
    print(f"\nPredicting {len(test_images)} images...")
    
    results = []
    for img_path in test_images:
        result = predict_image(img_path, model_data, verbose=False)
        if result:
            results.append((img_path, result))
    
    if not results:
        print("❌ No successful predictions")
        return
    
    # Calculate accuracy
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print('='*70)
    
    correct = 0
    for img_path, result in results:
        actual_label = 0 if 'benign' in img_path.lower() else 1
        predicted_label = result['class_label']
        is_correct = (actual_label == predicted_label)
        
        if is_correct:
            correct += 1
        
        symbol = '✅' if is_correct else '❌'
        actual_name = 'Benign' if actual_label == 0 else 'Malignant'
        
        print(f"{symbol} Actual: {actual_name:10s} | "
              f"Predicted: {result['class_name']:10s} | "
              f"Confidence: {result['confidence']:5.1f}%")
    
    accuracy = correct / len(results) * 100
    print(f"\n🎯 Validation Accuracy: {correct}/{len(results)} = {accuracy:.1f}%")
    
    if accuracy >= 90:
        print("✅✅✅ Excellent performance on validation set!")
    elif accuracy >= 80:
        print("✅✅ Good performance on validation set!")
    else:
        print("⚠️  Consider reviewing mislabeled images")

def interactive_mode(model_data):
    """Interactive prediction mode"""
    
    print(f"\n{'='*70}")
    print("INTERACTIVE PREDICTION MODE")
    print('='*70)
    print("\nOptions:")
    print("  1. Predict single image")
    print("  2. Predict all images in folder")
    print("  3. Test on validation samples")
    print("  4. Exit")
    
    while True:
        print("\n" + "-"*70)
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            img_path = input("Enter image path: ").strip()
            if img_path:
                predict_image(img_path, model_data)
        
        elif choice == '2':
            folder_path = input("Enter folder path: ").strip()
            if folder_path:
                predict_folder(folder_path, model_data)
        
        elif choice == '3':
            test_on_validation_set(model_data)
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid option. Please select 1-4.")

def main():
    parser = argparse.ArgumentParser(description='Breast Cancer Detection - Advanced Prediction')
    parser.add_argument('--model', type=str, default='models/svm_model_latest.pkl',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--test', action='store_true', 
                       help='Test on validation samples')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BREAST CANCER DETECTION - ADVANCED PREDICTION")
    print("="*70)
    
    # Load model
    print("\n1. Loading trained model...")
    model_data = load_trained_model(args.model)
    
    # Execute based on arguments
    if args.image:
        predict_image(args.image, model_data)
    
    elif args.folder:
        predict_folder(args.folder, model_data)
    
    elif args.test:
        test_on_validation_set(model_data)
    
    elif args.interactive:
        interactive_mode(model_data)
    
    else:
        # Default: show options and run interactive
        print("\n2. Select prediction mode:")
        print("\n" + "-"*70)
        print("Usage examples:")
        print("  python predict_advanced.py --image test_images/sample.png")
        print("  python predict_advanced.py --folder test_images/")
        print("  python predict_advanced.py --test")
        print("  python predict_advanced.py --interactive")
        print("-"*70)
        
        interactive_mode(model_data)
    
    print("\n" + "="*70)
    print("Prediction complete!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)