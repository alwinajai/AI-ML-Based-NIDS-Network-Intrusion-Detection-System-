"""
Preprocessing script for CSE-CIC-IDS2018-v3 dataset
Updated to find your actual data location
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import glob

def find_cic_ids_files():
    """
    Find CSE-CIC-IDS2018-v3 dataset files in the project
    """
    # Common patterns for CSE-CIC-IDS2018 files
    patterns = [
        "**/*cic*ids*2018*.csv",
        "**/*cic*2018*.csv", 
        "**/*ids*2018*.csv",
        "**/*traffic*.csv",
        "**/*network*.csv",
        "**/*.csv"
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(pattern, recursive=True))
    
    # Filter out very small files and focus on large datasets
    valid_files = []
    for file in found_files:
        try:
            size_mb = Path(file).stat().st_size / (1024**2)
            if size_mb > 1:  # Only files larger than 1MB
                valid_files.append((file, size_mb))
        except:
            continue
    
    # Sort by size (largest first)
    valid_files.sort(key=lambda x: x[1], reverse=True)
    
    return valid_files

def preprocess_cic_ids_data():
    """
    Preprocess CSE-CIC-IDS2018-v3 dataset for autoencoder
    """
    print("Looking for CSE-CIC-IDS2018-v3 dataset files...")
    
    # Find potential dataset files
    found_files = find_cic_ids_files()
    
    if not found_files:
        print("❌ No CSV files found in project directory!")
        print("Please place your CSE-CIC-IDS2018-v3 dataset in the project folder")
        return False
    
    print(f"\nFound {len(found_files)} potential dataset files:")
    for i, (file, size_mb) in enumerate(found_files[:10], 1):
        print(f"{i:2d}. {file} ({size_mb:.1f} MB)")
    
    # Ask user to select the correct file
    if len(found_files) == 1:
        selected_file = found_files[0][0]
        print(f"\nUsing: {selected_file}")
    else:
        choice = input(f"\nSelect file number (1-{len(found_files)}): ")
        try:
            choice_idx = int(choice) - 1
            selected_file = found_files[choice_idx][0]
        except:
            print("Invalid selection, using first file")
            selected_file = found_files[0][0]
    
    print(f"\nLoading dataset: {selected_file}")
    
    try:
        # Load the dataset
        df = pd.read_csv(selected_file)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Look for common label column names in CSE-CIC-IDS2018
        label_columns = ['Label', 'label', 'class', 'Class', 'Attack', 'Category', 'category']
        label_col = None
        
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        if label_col:
            print(f"Found label column: '{label_col}'")
            labels = df[label_col].copy()
            features_df = df.drop(label_col, axis=1)
        else:
            print("No standard label column found, treating all as normal traffic")
            labels = pd.Series([0] * len(df))
            features_df = df.copy()
        
        print(f"Features shape: {features_df.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Basic preprocessing
        print("\nPreprocessing data...")
        
        # Keep only numeric columns for autoencoder
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        features_df = features_df[numeric_cols]
        
        print(f"After keeping only numeric columns: {features_df.shape}")
        
        # Handle missing values
        features_df = features_df.fillna(features_df.mean())
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.mean())
        
        # Create directories
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_df.to_csv("data/processed/cic_ids_features.csv", index=False)
        print(f"✓ Features saved to data/processed/cic_ids_features.csv")
        
        # Process labels (convert to binary: 0=normal, 1=attack)
        if label_col:
            # Convert to binary classification
            unique_labels = labels.unique()
            print(f"Unique labels: {unique_labels}")
            
            # Identify normal traffic labels (common in CSE-CIC-IDS2018)
            normal_labels = ['BENIGN', 'Normal', 'normal', 'benign', 'LEGITIMATE', 'Legitimate']
            normal_traffic = [label for label in unique_labels if str(label).upper() in [nl.upper() for nl in normal_labels]]
            
            if normal_traffic:
                normal_label = normal_traffic[0]
                print(f"Normal traffic label: '{normal_label}'")
                labels_binary = (labels != normal_label).astype(int)
            else:
                # If no clear normal label found, assume first unique value is normal
                normal_label = unique_labels[0]
                print(f"No clear normal label found, assuming '{normal_label}' is normal")
                labels_binary = (labels != normal_label).astype(int)
            
            labels_df = pd.DataFrame({'label': labels_binary})
            labels_df.to_csv("data/processed/cic_ids_labels.csv", index=False)
            print(f"✓ Labels saved to data/processed/cic_ids_labels.csv")
            print(f"Normal traffic: {np.sum(labels_binary == 0)} samples")
            print(f"Attack traffic: {np.sum(labels_binary == 1)} samples")
        else:
            # Create dummy labels (all normal)
            labels_df = pd.DataFrame({'label': [0] * len(features_df)})
            labels_df.to_csv("data/processed/cic_ids_labels.csv", index=False)
            print("✓ Dummy labels created (all normal traffic)")
        
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"Features: data/processed/cic_ids_features.csv")
        print(f"Labels: data/processed/cic_ids_labels.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    preprocess_cic_ids_data()