import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70)

def print_section(text):
    """Print section header"""
    print("\n" + text)
    print("-"*70)

def test_python_version():
    """Test Python version"""
    print_section("1. Testing Python Version")
    
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python version is compatible (3.8+)")
        return True
    else:
        print("   ❌ Python 3.8 or higher required")
        print("   Please update Python: https://www.python.org/downloads/")
        return False

def test_imports():
    """Test if all required packages are installed"""
    print_section("2. Testing Package Imports")
    
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'opencv-python': 'cv2',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm',
        'pillow': 'PIL',
        'pydicom': 'pydicom'
    }
    
    all_ok = True
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - NOT INSTALLED")
            all_ok = False
    
    if all_ok:
        print("\n   ✅ All packages are installed!")
    else:
        print("\n   ❌ Some packages are missing")
        print("   Install with: pip install -r requirements.txt")
    
    return all_ok

def test_folder_structure():
    """Test if required folders exist"""
    print_section("3. Testing Folder Structure")
    
    required_folders = [
        ('raw_dataset', False, 'Place downloaded dataset here'),
        ('organized_dataset', False, 'Created after running organize_dataset.py'),
        ('models', False, 'Created automatically during training'),
        ('results', False, 'Created automatically during training'),
        ('test_images', False, 'Optional: Add test images here')
    ]
    
    for folder, is_required, description in required_folders:
        exists = os.path.exists(folder)
        
        if exists:
            # Count files
            file_count = sum([len(files) for r, d, files in os.walk(folder)])
            print(f"   ✅ {folder}/ ({file_count} files)")
        else:
            symbol = '❌' if is_required else '⚠️ '
            print(f"   {symbol} {folder}/ - NOT FOUND")
            print(f"      {description}")
    
    return True

def test_script_files():
    """Test if all required Python scripts exist"""
    print_section("4. Testing Script Files")
    
    scripts = [
        ('dataset_organizer.py', True, 'Dataset organization functions'),
        ('breast_cancer_svm.py', True, 'Main SVM model'),
        ('organize_dataset.py', True, 'Quick organization script'),
        ('train_model.py', True, 'Model training script'),
        ('predict.py', True, 'Prediction script'),
        ('requirements.txt', True, 'Python dependencies'),
    ]
    
    all_ok = True
    for script, is_required, description in scripts:
        exists = os.path.exists(script)
        
        if exists:
            size = os.path.getsize(script)
            print(f"   ✅ {script} ({size} bytes)")
        else:
            if is_required:
                print(f"   ❌ {script} - NOT FOUND")
                print(f"      {description}")
                all_ok = False
            else:
                print(f"   ⚠️  {script} - OPTIONAL")
    
    return all_ok

def test_dataset():
    """Test if dataset is organized"""
    print_section("5. Testing Dataset")
    
    benign_path = 'organized_dataset/benign'
    malignant_path = 'organized_dataset/malignant'
    
    if not os.path.exists('organized_dataset'):
        print("   ⚠️  Dataset not organized yet")
        print("      Run: python organize_dataset.py")
        return False
    
    if not os.path.exists(benign_path) or not os.path.exists(malignant_path):
        print("   ⚠️  benign/ or malignant/ folders not found")
        print("      Run: python organize_dataset.py")
        return False
    
    # Count images
    benign_files = [f for f in os.listdir(benign_path) 
                   if os.path.isfile(os.path.join(benign_path, f))]
    malignant_files = [f for f in os.listdir(malignant_path) 
                      if os.path.isfile(os.path.join(malignant_path, f))]
    
    benign_count = len(benign_files)
    malignant_count = len(malignant_files)
    total = benign_count + malignant_count
    
    if total == 0:
        print("   ❌ No images found in dataset folders")
        print("      Run: python organize_dataset.py")
        return False
    
    print(f"   ✅ Dataset organized successfully!")
    print(f"      Benign images: {benign_count}")
    print(f"      Malignant images: {malignant_count}")
    print(f"      Total: {total}")
    
    # Check balance
    if benign_count > 0 and malignant_count > 0:
        ratio = max(benign_count, malignant_count) / min(benign_count, malignant_count)
        if ratio > 3:
            print(f"      ⚠️  Dataset is imbalanced (ratio: {ratio:.1f}:1)")
            print(f"         Consider balancing the dataset")
        else:
            print(f"      ✅ Dataset is reasonably balanced")
    
    # Check minimum size
    if total < 50:
        print(f"      ⚠️  Dataset is small (<50 images)")
        print(f"         Model accuracy may be limited")
    elif total < 200:
        print(f"      ⚠️  Dataset is moderate (50-200 images)")
        print(f"         More data recommended for better accuracy")
    else:
        print(f"      ✅ Dataset size is good (>200 images)")
    
    return True

def test_trained_model():
    """Test if model is trained"""
    print_section("6. Testing Trained Model")
    
    model_path = 'models/svm_model_latest.pkl'
    
    if not os.path.exists('models'):
        print("   ⚠️  No models folder")
        print("      Train model: python train_model.py")
        return False
    
    if not os.path.exists(model_path):
        print("   ⚠️  No trained model found")
        print("      Train model: python train_model.py")
        return False
    
    # Check model file
    size = os.path.getsize(model_path)
    print(f"   ✅ Trained model found!")
    print(f"      File: {model_path}")
    print(f"      Size: {size:,} bytes")
    
    # Try loading model
    try:
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            print(f"      Accuracy: {metrics['accuracy']*100:.2f}%")
            if metrics['accuracy'] >= 0.85:
                print(f"      ✅ Model meets 85% accuracy requirement!")
            else:
                print(f"      ⚠️  Model accuracy below 85%")
        
        if 'training_date' in model_data:
            print(f"      Trained: {model_data['training_date']}")
        
    except Exception as e:
        print(f"      ⚠️  Could not load model details: {str(e)}")
    
    return True

def generate_next_steps(python_ok, imports_ok, scripts_ok, dataset_ok, model_ok):
    """Generate personalized next steps"""
    print_header("NEXT STEPS")
    
    if not python_ok:
        print("\n📝 Step 1: Update Python")
        print("   Download Python 3.8+: https://www.python.org/downloads/")
        return
    
    if not imports_ok:
        print("\n📝 Step 1: Install Dependencies")
        print("   Run: pip install -r requirements.txt")
        return
    
    if not scripts_ok:
        print("\n📝 Step 1: Create Missing Scripts")
        print("   Copy the provided scripts to your project folder")
        return
    
    if not dataset_ok:
        print("\n📝 Step 1: Download Dataset")
        print("   Download from:")
        print("   - CBIS-DDSM: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset")
        print("   - MIAS: https://www.kaggle.com/datasets/kmader/mias-mammography")
        print("\n📝 Step 2: Place Dataset")
        print("   Extract and place in: raw_dataset/")
        print("\n📝 Step 3: Organize Dataset")
        print("   Run: python organize_dataset.py")
        return
    
    if not model_ok:
        print("\n📝 Step 1: Train Model")
        print("   Run: python train_model.py")
        print("   (This will take 5-15 minutes)")
        print("\n📝 Step 2: Make Predictions")
        print("   Run: python predict.py")
        return
    
    # All setup complete!
    print("\n✅ Your setup is COMPLETE and ready to use!")
    print("\n📝 What you can do now:")
    print("   1. Test predictions:")
    print("      python predict.py --test")
    print("\n   2. Predict single image:")
    print("      python predict.py --image path/to/image.png")
    print("\n   3. Predict folder:")
    print("      python predict.py --folder test_images/")
    print("\n   4. Interactive mode:")
    print("      python predict.py --interactive")
    print("\n   5. Retrain with different settings:")
    print("      Edit breast_cancer_svm.py and run train_model.py")

def main():
    """Main testing function"""
    print_header("BREAST CANCER DETECTION - SETUP TEST")
    
    print("\nThis script will verify your project setup is correct.")
    print("Please wait while we check all components...")
    
    # Run all tests
    python_ok = test_python_version()
    imports_ok = test_imports()
    test_folder_structure()  # Always show, not critical
    scripts_ok = test_script_files()
    dataset_ok = test_dataset()
    model_ok = test_trained_model()
    
    # Summary
    print_header("SETUP TEST SUMMARY")
    
    print("\nStatus:")
    print(f"   {'✅' if python_ok else '❌'} Python 3.8+")
    print(f"   {'✅' if imports_ok else '❌'} Required packages")
    print(f"   {'✅' if scripts_ok else '❌'} Python scripts")
    print(f"   {'✅' if dataset_ok else '⚠️ '} Dataset organized")
    print(f"   {'✅' if model_ok else '⚠️ '} Model trained")
    
    # Generate next steps
    generate_next_steps(python_ok, imports_ok, scripts_ok, dataset_ok, model_ok)
    
    print("\n" + "="*70)
    
    # Return exit code
    if python_ok and imports_ok and scripts_ok:
        return 0
    else:
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)