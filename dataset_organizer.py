"""
MAMMOGRAM DATASET ORGANIZER
============================
This script helps organize large mammogram datasets into benign/malignant folders.

Supports:
- CBIS-DDSM dataset (CSV with metadata)
- MIAS dataset (text file annotations)
- Folder structures with naming conventions
- Manual CSV file with image paths and labels

Author: Dataset Organization Tool
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import re
from tqdm import tqdm
import json

class MammogramDatasetOrganizer:
    """
    Organize mammogram datasets into benign/malignant structure
    """
    
    def __init__(self, source_dir, output_dir='organized_dataset'):
        """
        Parameters:
        -----------
        source_dir : str
            Path to source dataset directory
        output_dir : str
            Path where organized dataset will be created
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.benign_dir = self.output_dir / 'benign'
        self.malignant_dir = self.output_dir / 'malignant'
        
        self.benign_dir.mkdir(parents=True, exist_ok=True)
        self.malignant_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_processed': 0,
            'benign_count': 0,
            'malignant_count': 0,
            'skipped': 0,
            'errors': []
        }
    
    def organize_cbis_ddsm(self, csv_file=None, image_column='image file path', 
                          label_column='pathology'):
        """
        Organize CBIS-DDSM dataset using CSV metadata
        
        Parameters:
        -----------
        csv_file : str
            Path to CSV file with metadata (e.g., 'mass_case_description_train_set.csv')
        image_column : str
            Column name containing image paths
        label_column : str
            Column name containing pathology labels
        """
        print("\n" + "="*70)
        print("ORGANIZING CBIS-DDSM DATASET")
        print("="*70)
        
        if csv_file is None:
            # Try to find CSV files automatically
            csv_files = list(self.source_dir.glob('*.csv'))
            if not csv_files:
                print("❌ No CSV files found. Please provide csv_file parameter.")
                return
            csv_file = csv_files[0]
            print(f"Found CSV file: {csv_file}")
        
        # Read CSV
        df = pd.read_csv(csv_file)
        print(f"\n📊 Dataset Info:")
        print(f"   Total records: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check if columns exist
        if label_column not in df.columns:
            print(f"\n⚠️  Column '{label_column}' not found. Available columns:")
            print(df.columns.tolist())
            # Try to find pathology column
            path_cols = [col for col in df.columns if 'path' in col.lower()]
            if path_cols:
                label_column = path_cols[0]
                print(f"Using column: '{label_column}'")
        
        if image_column not in df.columns:
            img_cols = [col for col in df.columns if 'image' in col.lower() or 'file' in col.lower()]
            if img_cols:
                image_column = img_cols[0]
                print(f"Using image column: '{image_column}'")
        
        print(f"\n🔍 Processing images...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
            try:
                img_path = row[image_column]
                pathology = str(row[label_column]).upper()
                
                # Determine if benign or malignant
                if 'BENIGN' in pathology:
                    dest_dir = self.benign_dir
                    self.stats['benign_count'] += 1
                elif 'MALIGNANT' in pathology:
                    dest_dir = self.malignant_dir
                    self.stats['malignant_count'] += 1
                else:
                    self.stats['skipped'] += 1
                    continue
                
                # Find and copy image
                full_path = self.source_dir / img_path
                if not full_path.exists():
                    # Try alternative paths
                    full_path = self.source_dir / Path(img_path).name
                
                if full_path.exists():
                    dest_path = dest_dir / f"{full_path.stem}_{idx}{full_path.suffix}"
                    shutil.copy2(full_path, dest_path)
                    self.stats['total_processed'] += 1
                else:
                    self.stats['skipped'] += 1
                    
            except Exception as e:
                self.stats['errors'].append(f"Row {idx}: {str(e)}")
                continue
        
        self.print_summary()
    
    def organize_mias_dataset(self, info_file='Info.txt'):
        """
        Organize MIAS dataset using Info.txt file
        
        MIAS format: Each line contains image info with pathology code
        B = Benign, M = Malignant
        """
        print("\n" + "="*70)
        print("ORGANIZING MIAS DATASET")
        print("="*70)
        
        info_path = self.source_dir / info_file
        if not info_path.exists():
            print(f"❌ {info_file} not found in {self.source_dir}")
            return
        
        print(f"Reading {info_file}...")
        
        with open(info_path, 'r') as f:
            lines = f.readlines()
        
        print(f"Found {len(lines)} records")
        print("\n🔍 Processing images...")
        
        for line in tqdm(lines, desc="Organizing"):
            try:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                image_name = parts[0]
                severity = parts[2]  # B=Benign, M=Malignant
                
                # Find image file
                image_file = None
                for ext in ['.pgm', '.png', '.jpg', '.jpeg']:
                    potential_path = self.source_dir / f"{image_name}{ext}"
                    if potential_path.exists():
                        image_file = potential_path
                        break
                
                if image_file is None:
                    self.stats['skipped'] += 1
                    continue
                
                # Determine destination
                if severity == 'B':
                    dest_dir = self.benign_dir
                    self.stats['benign_count'] += 1
                elif severity == 'M':
                    dest_dir = self.malignant_dir
                    self.stats['malignant_count'] += 1
                else:
                    self.stats['skipped'] += 1
                    continue
                
                # Copy file
                dest_path = dest_dir / image_file.name
                shutil.copy2(image_file, dest_path)
                self.stats['total_processed'] += 1
                
            except Exception as e:
                self.stats['errors'].append(f"Line '{line.strip()}': {str(e)}")
                continue
        
        self.print_summary()
    
    def organize_from_custom_csv(self, csv_file, image_path_col='image_path', 
                                 label_col='label'):
        """
        Organize dataset from custom CSV file
        
        CSV format:
        image_path,label
        path/to/image1.png,benign
        path/to/image2.png,malignant
        
        Parameters:
        -----------
        csv_file : str
            Path to CSV file
        image_path_col : str
            Column name for image paths
        label_col : str
            Column name for labels (benign/malignant or 0/1)
        """
        print("\n" + "="*70)
        print("ORGANIZING FROM CUSTOM CSV")
        print("="*70)
        
        df = pd.read_csv(csv_file)
        print(f"\n📊 Dataset Info:")
        print(f"   Total records: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        print("\n🔍 Processing images...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
            try:
                img_path = row[image_path_col]
                label = str(row[label_col]).lower()
                
                # Determine destination
                if label in ['benign', 'b', '0', 'normal']:
                    dest_dir = self.benign_dir
                    self.stats['benign_count'] += 1
                elif label in ['malignant', 'm', '1', 'cancer']:
                    dest_dir = self.malignant_dir
                    self.stats['malignant_count'] += 1
                else:
                    self.stats['skipped'] += 1
                    continue
                
                # Find image
                full_path = Path(img_path)
                if not full_path.is_absolute():
                    full_path = self.source_dir / img_path
                
                if full_path.exists():
                    dest_path = dest_dir / f"{full_path.stem}_{idx}{full_path.suffix}"
                    shutil.copy2(full_path, dest_path)
                    self.stats['total_processed'] += 1
                else:
                    self.stats['skipped'] += 1
                    
            except Exception as e:
                self.stats['errors'].append(f"Row {idx}: {str(e)}")
                continue
        
        self.print_summary()
    
    def organize_from_naming_convention(self, benign_keywords=['benign', 'normal'], 
                                       malignant_keywords=['malignant', 'cancer']):
        """
        Organize images based on filename keywords
        
        Parameters:
        -----------
        benign_keywords : list
            Keywords in filename that indicate benign
        malignant_keywords : list
            Keywords in filename that indicate malignant
        """
        print("\n" + "="*70)
        print("ORGANIZING BY FILENAME KEYWORDS")
        print("="*70)
        print(f"Benign keywords: {benign_keywords}")
        print(f"Malignant keywords: {malignant_keywords}")
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.dcm', '.pgm', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.source_dir.rglob(f'*{ext}'))
        
        print(f"\nFound {len(image_files)} image files")
        print("\n🔍 Processing images...")
        
        for img_file in tqdm(image_files, desc="Organizing"):
            try:
                filename_lower = img_file.name.lower()
                
                # Check for keywords
                is_benign = any(keyword in filename_lower for keyword in benign_keywords)
                is_malignant = any(keyword in filename_lower for keyword in malignant_keywords)
                
                if is_benign and not is_malignant:
                    dest_dir = self.benign_dir
                    self.stats['benign_count'] += 1
                elif is_malignant and not is_benign:
                    dest_dir = self.malignant_dir
                    self.stats['malignant_count'] += 1
                else:
                    self.stats['skipped'] += 1
                    continue
                
                # Copy file
                dest_path = dest_dir / img_file.name
                if dest_path.exists():
                    # Add counter if file exists
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_dir / f"{img_file.stem}_{counter}{img_file.suffix}"
                        counter += 1
                
                shutil.copy2(img_file, dest_path)
                self.stats['total_processed'] += 1
                
            except Exception as e:
                self.stats['errors'].append(f"File {img_file}: {str(e)}")
                continue
        
        self.print_summary()
    
    def organize_from_folder_structure(self, benign_folder_names=['benign', 'normal'],
                                       malignant_folder_names=['malignant', 'cancer']):
        """
        Organize when images are already in subfolders
        
        Example structure:
        source_dir/
        ├── benign_cases/
        ├── malignant_cases/
        └── ...
        """
        print("\n" + "="*70)
        print("ORGANIZING FROM FOLDER STRUCTURE")
        print("="*70)
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.dcm', '.pgm', '.tif', '.tiff']
        
        # Find all subdirectories
        subdirs = [d for d in self.source_dir.iterdir() if d.is_dir()]
        print(f"Found {len(subdirs)} subdirectories")
        
        for subdir in subdirs:
            folder_name_lower = subdir.name.lower()
            
            # Determine if folder contains benign or malignant
            is_benign = any(keyword in folder_name_lower for keyword in benign_folder_names)
            is_malignant = any(keyword in folder_name_lower for keyword in malignant_folder_names)
            
            if not (is_benign or is_malignant):
                print(f"⚠️  Skipping folder: {subdir.name}")
                continue
            
            dest_dir = self.benign_dir if is_benign else self.malignant_dir
            label = "benign" if is_benign else "malignant"
            
            # Get all images from this folder
            image_files = []
            for ext in image_extensions:
                image_files.extend(subdir.rglob(f'*{ext}'))
            
            print(f"\n📁 Processing '{subdir.name}' ({label}): {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=f"Copying {label}"):
                try:
                    dest_path = dest_dir / img_file.name
                    if dest_path.exists():
                        counter = 1
                        while dest_path.exists():
                            dest_path = dest_dir / f"{img_file.stem}_{counter}{img_file.suffix}"
                            counter += 1
                    
                    shutil.copy2(img_file, dest_path)
                    self.stats['total_processed'] += 1
                    
                    if is_benign:
                        self.stats['benign_count'] += 1
                    else:
                        self.stats['malignant_count'] += 1
                        
                except Exception as e:
                    self.stats['errors'].append(f"File {img_file}: {str(e)}")
                    continue
        
        self.print_summary()
    
    def create_metadata_csv(self):
        """
        Create a CSV file with metadata of organized dataset
        """
        metadata = []
        
        # Process benign images
        for img_file in self.benign_dir.iterdir():
            if img_file.is_file():
                metadata.append({
                    'filename': img_file.name,
                    'path': str(img_file),
                    'label': 'benign',
                    'class': 0
                })
        
        # Process malignant images
        for img_file in self.malignant_dir.iterdir():
            if img_file.is_file():
                metadata.append({
                    'filename': img_file.name,
                    'path': str(img_file),
                    'label': 'malignant',
                    'class': 1
                })
        
        # Save to CSV
        df = pd.DataFrame(metadata)
        csv_path = self.output_dir / 'dataset_metadata.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Metadata CSV created: {csv_path}")
        print(f"   Total entries: {len(df)}")
    
    def print_summary(self):
        """
        Print organization summary
        """
        print("\n" + "="*70)
        print("ORGANIZATION SUMMARY")
        print("="*70)
        print(f"✅ Successfully processed: {self.stats['total_processed']} images")
        print(f"   Benign images:     {self.stats['benign_count']}")
        print(f"   Malignant images:  {self.stats['malignant_count']}")
        print(f"⚠️  Skipped:           {self.stats['skipped']} images")
        
        if self.stats['errors']:
            print(f"❌ Errors:            {len(self.stats['errors'])}")
            print("\nFirst 5 errors:")
            for error in self.stats['errors'][:5]:
                print(f"   - {error}")
        
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"   Benign folder:  {self.benign_dir}")
        print(f"   Malignant folder: {self.malignant_dir}")
        
        # Create metadata
        self.create_metadata_csv()
        
        print("\n" + "="*70)
        print("✨ ORGANIZATION COMPLETE!")
        print("="*70)
    
    def auto_detect_and_organize(self):
        """
        Automatically detect dataset type and organize
        """
        print("\n" + "="*70)
        print("AUTO-DETECTING DATASET TYPE")
        print("="*70)
        
        # Check for CBIS-DDSM CSV files
        csv_files = list(self.source_dir.glob('*.csv'))
        if csv_files:
            print("✓ Found CSV files - attempting CBIS-DDSM format")
            self.organize_cbis_ddsm(csv_files[0])
            return
        
        # Check for MIAS Info.txt
        if (self.source_dir / 'Info.txt').exists():
            print("✓ Found Info.txt - attempting MIAS format")
            self.organize_mias_dataset()
            return
        
        # Check for existing folder structure
        subdirs = [d for d in self.source_dir.iterdir() if d.is_dir()]
        benign_folders = [d for d in subdirs if 'benign' in d.name.lower() or 'normal' in d.name.lower()]
        malignant_folders = [d for d in subdirs if 'malignant' in d.name.lower() or 'cancer' in d.name.lower()]
        
        if benign_folders or malignant_folders:
            print("✓ Found benign/malignant folders - organizing from folder structure")
            self.organize_from_folder_structure()
            return
        
        # Try filename keywords as last resort
        print("⚠️  No standard format detected - trying filename keywords")
        self.organize_from_naming_convention()


# ============================================================================
# EASY-TO-USE FUNCTIONS
# ============================================================================

def organize_cbis_ddsm(source_dir, output_dir='organized_dataset', csv_file=None):
    """Quick function for CBIS-DDSM dataset"""
    organizer = MammogramDatasetOrganizer(source_dir, output_dir)
    organizer.organize_cbis_ddsm(csv_file)

def organize_mias(source_dir, output_dir='organized_dataset'):
    """Quick function for MIAS dataset"""
    organizer = MammogramDatasetOrganizer(source_dir, output_dir)
    organizer.organize_mias_dataset()

def organize_from_csv(source_dir, csv_file, output_dir='organized_dataset'):
    """Quick function for custom CSV"""
    organizer = MammogramDatasetOrganizer(source_dir, output_dir)
    organizer.organize_from_custom_csv(csv_file)

def auto_organize(source_dir, output_dir='organized_dataset'):
    """Auto-detect and organize any dataset"""
    organizer = MammogramDatasetOrganizer(source_dir, output_dir)
    organizer.auto_detect_and_organize()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    USAGE EXAMPLES:
    ---------------
    
    # Example 1: Auto-detect dataset type
    auto_organize(
        source_dir='path/to/downloaded/dataset',
        output_dir='organized_dataset'
    )
    
    # Example 2: CBIS-DDSM with CSV
    organize_cbis_ddsm(
        source_dir='path/to/cbis_ddsm',
        csv_file='mass_case_description_train_set.csv',
        output_dir='organized_dataset'
    )
    
    # Example 3: MIAS dataset
    organize_mias(
        source_dir='path/to/mias',
        output_dir='organized_dataset'
    )
    
    # Example 4: Custom CSV
    organize_from_csv(
        source_dir='path/to/images',
        csv_file='labels.csv',
        output_dir='organized_dataset'
    )
    
    # Example 5: Manual organization with full control
    organizer = MammogramDatasetOrganizer(
        source_dir='path/to/dataset',
        output_dir='my_organized_data'
    )
    
    # Choose one method:
    organizer.organize_cbis_ddsm()
    # organizer.organize_mias_dataset()
    # organizer.organize_from_naming_convention()
    # organizer.organize_from_folder_structure()
    # organizer.auto_detect_and_organize()
    """
    
    print("\n" + "="*70)
    print("MAMMOGRAM DATASET ORGANIZER")
    print("="*70)
    print("\n📋 This script helps organize large mammogram datasets.")
    print("\nTo use this script, edit the code below with your paths:")
    print("\n" + "-"*70)
    print("EXAMPLE USAGE:")
    print("-"*70)
    print("""
    # Option 1: Auto-detect (Recommended)
    auto_organize(
        source_dir='path/to/your/downloaded/dataset',
        output_dir='organized_dataset'
    )
    
    # Option 2: Specific dataset type
    organizer = MammogramDatasetOrganizer(
        source_dir='path/to/dataset',
        output_dir='organized_dataset'
    )
    organizer.auto_detect_and_organize()
    """)
    print("-"*70)
    
    # UNCOMMENT AND MODIFY THE LINES BELOW TO USE:
    
    # auto_organize(
    #     source_dir='path/to/your/dataset',
    #     output_dir='organized_dataset'
    # )