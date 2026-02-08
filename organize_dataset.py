"""
QUICK DATASET ORGANIZATION SCRIPT
==================================
Run this script to organize your downloaded mammogram dataset
into benign and malignant folders.

Usage:
    python organize_dataset.py
"""

from dataset_organizer import auto_organize
import os
import sys

def main():
    print("\n" + "="*70)
    print("BREAST CANCER DATASET ORGANIZER")
    print("="*70)
    
    # Configuration
    SOURCE_DIR = 'raw_dataset'  # Change this if your dataset is elsewhere
    OUTPUT_DIR = 'organized_dataset'
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"\n❌ Error: '{SOURCE_DIR}' folder not found!")
        print("\n📝 Please follow these steps:")
        print("   1. Create a folder named 'raw_dataset' in your project directory")
        print("   2. Download a mammogram dataset (CBIS-DDSM or MIAS)")
        print("   3. Extract and place all files in 'raw_dataset/' folder")
        print("   4. Run this script again")
        print("\n📚 Dataset links:")
        print("   - CBIS-DDSM: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset")
        print("   - MIAS: https://www.kaggle.com/datasets/kmader/mias-mammography")
        sys.exit(1)
    
    # Check if folder is empty
    if not os.listdir(SOURCE_DIR):
        print(f"\n⚠️  Warning: '{SOURCE_DIR}' folder is empty!")
        print("Please add your dataset files to this folder and run again.")
        sys.exit(1)
    
    print(f"\n✅ Found source directory: {SOURCE_DIR}")
    print(f"📁 Output will be saved to: {OUTPUT_DIR}")
    
    # Count files in source
    total_files = sum([len(files) for r, d, files in os.walk(SOURCE_DIR)])
    print(f"📊 Total files found: {total_files}")
    
    # Ask for confirmation
    print("\n" + "-"*70)
    response = input("Proceed with organization? (y/n): ").lower()
    
    if response != 'y':
        print("Organization cancelled.")
        sys.exit(0)
    
    print("\n" + "="*70)
    print("STARTING ORGANIZATION...")
    print("="*70)
    
    try:
        # Run auto-organization
        auto_organize(
            source_dir=SOURCE_DIR,
            output_dir=OUTPUT_DIR
        )
        
        print("\n" + "="*70)
        print("✅ ORGANIZATION COMPLETE!")
        print("="*70)
        print(f"\n📁 Your organized dataset is ready in: {OUTPUT_DIR}/")
        print("\nNext steps:")
        print("   1. Check the organized_dataset/ folder")
        print("   2. Verify benign/ and malignant/ folders have images")
        print("   3. Run: python train_model.py")
        
    except Exception as e:
        print(f"\n❌ Error during organization: {str(e)}")
        print("\nPossible solutions:")
        print("   1. Check if dataset files are valid")
        print("   2. Ensure you have read/write permissions")
        print("   3. Try running as administrator")
        sys.exit(1)

if __name__ == "__main__":
    main()