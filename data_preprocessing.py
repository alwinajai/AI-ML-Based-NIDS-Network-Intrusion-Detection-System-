# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder # Removed VarianceThreshold import

def load_data(raw_data_dir):
    """
    Loads the main CSV file (NF-CICIDS2018-v3.csv) from the raw data directory.
    """
    main_file_path = os.path.join(raw_data_dir, 'NF-CICIDS2018-v3.csv')
    print(f"Attempting to load file: {main_file_path}")

    # Check if the file exists first
    if not os.path.exists(main_file_path):
        print(f"ERROR: File {main_file_path} does not exist.")
        # List contents of the raw directory for debugging
        print(f"Contents of {raw_data_dir}:")
        try:
            print(os.listdir(raw_data_dir))
        except Exception as e:
            print(f"Could not list contents of {raw_data_dir}: {e}")
        raise FileNotFoundError(f"Main data file 'NF-CICIDS2018-v3.csv' not found in {raw_data_dir}")

    print(f"File {main_file_path} found. Size: {os.path.getsize(main_file_path)} bytes")
    print("Attempting to read CSV...")

    try:
        # The dataset often has issues with inconsistent numbers of fields. Using 'on_bad_lines' handles this.
        df = pd.read_csv(main_file_path, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"ERROR: Failed to parse CSV file {main_file_path}: {e}")
        raise e
    except UnicodeDecodeError as e:
        print(f"ERROR: Failed to decode CSV file {main_file_path} (likely encoding issue): {e}")
        # Try common encodings if the default fails
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        for enc in encodings_to_try:
            try:
                print(f"Trying encoding: {enc}")
                df = pd.read_csv(main_file_path, on_bad_lines='skip', encoding=enc)
                print(f"Successfully read with encoding: {enc}")
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        if df is None:
            raise FileNotFoundError(f"Could not read {main_file_path} with common encodings.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while reading {main_file_path}: {e}")
        raise e

    print(f"Successfully loaded data. Shape: {df.shape}")
    return df

def explore_data(df):
    """
    Performs basic exploration of the loaded DataFrame.
    """
    print("\n--- Dataset Shape ---")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\n--- Column Names (first 10) ---")
    print(df.columns.tolist()[:10]) # Print first 10, there might be many
    print("--- Column Names (last 10) ---")
    print(df.columns.tolist()[-10:]) # Print last 10

    print("\n--- Data Types (first 10) ---")
    print(df.dtypes[:10])
    print("--- Data Types (last 10) ---")
    print(df.dtypes[-10:])

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Missing Values (Top 10) ---")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\n--- Target Variable Distribution (Label) ---")
    if 'Label' in df.columns:
        print(df['Label'].value_counts())
    else:
        print("Column 'Label' not found. Please check the target variable name in your dataset.")
        # Print all column names containing 'label' or 'attack' (case-insensitive)
        label_like_cols = [col for col in df.columns if 'label' in col.lower() or 'attack' in col.lower()]
        print(f"Possible label columns: {label_like_cols}")
        if label_like_cols:
            print(f"Example distribution for '{label_like_cols[0]}':")
            print(df[label_like_cols[0]].value_counts())


def basic_preprocessing(df):
    """
    Applies basic preprocessing steps specific to the NF-CSE-CIC-IDS2018-v3 dataset.
    Uses memory-efficient techniques for large datasets.
    """
    print("\n--- Starting Basic Preprocessing (Memory Efficient) ---")
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")

    # 1. Drop columns with too many missing values (e.g., > 50%)
    # This step should be fine as it's row-wise, not column-wise aggregation
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    cols_to_drop_na = missing_percentages[missing_percentages > 50].index.tolist()
    if cols_to_drop_na:
        print(f"Dropping columns with > 50% missing values: {cols_to_drop_na}")
        df = df.drop(columns=cols_to_drop_na)
    else:
        print("No columns found with > 50% missing values.")

    # 2. Identify and drop columns that are likely identifiers or timestamps
    # Usually 'Flow ID' and 'Timestamp' are safe to drop.
    # Note: FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS are present, might be dropped later if needed for live capture features
    id_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['src ip', 'dst ip', 'simillar_magnitude', 'sflow_fpi', 'dflow_fpi'])]
    if id_cols:
        print(f"Dropping identifier columns: {id_cols}")
        df = df.drop(columns=id_cols)
    else:
        print("No obvious identifier columns found to drop based on common names.")

    # Separate numerical and categorical columns *after* dropping ID columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    print(f"Identified {len(numeric_cols)} numerical columns and {len(categorical_cols)} categorical columns.")

    # 3. Handle missing values efficiently
    # For numerical columns: Calculate median per column using a memory-efficient method (iterating over columns)
    print("Handling missing values in numerical columns...")
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median() # This calculates median for one column at a time, much more memory efficient
            df[col] = df[col].fillna(median_val)
            print(f"  Filled missing values in '{col}' with median: {median_val}")

    # For categorical columns: Calculate mode per column (iterating over columns)
    print("Handling missing values in categorical columns...")
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_series = df[col].mode() # This returns a Series, take the first value if it exists
            if not mode_series.empty:
                mode_val = mode_series[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  Filled missing values in '{col}' with mode: {mode_val}")
            else:
                # If mode() is empty (e.g., all values are NaN or unique), fill with a default like 'Unknown'
                df[col] = df[col].fillna('Unknown')
                print(f"  Filled missing values in '{col}' with default: 'Unknown'")

    # 4. Check for infinite values and replace with NaN, then handle them again (using the efficient method)
    print("Replacing infinite values...")
    df = df.replace([np.inf, -np.inf], np.nan)
    # Re-run the imputation for any newly introduced NaNs from inf values using the efficient loop method
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_series = df[col].mode()
            if not mode_series.empty:
                mode_val = mode_series[0]
                df[col] = df[col].fillna(mode_val)
            else:
                df[col] = df[col].fillna('Unknown')

    # 5. Identify constant or quasi-constant features (very low variance) - Numerical
    # SKIPPED for now due to memory constraints with large dataset
    # A full variance check across all rows is too expensive.
    # Could be done later on a sample if needed.
    print("Skipping constant/quasi-constant feature check due to memory constraints on full dataset.")

    # 6. Encode the Target Variable ('Label') if it's categorical
    label_col = 'Label' # Adjust if exploration showed a different name
    if label_col in df.columns and df[label_col].dtype == 'object':
        print(f"Encoding target variable '{label_col}'...")
        le = LabelEncoder()
        df[label_col] = le.fit_transform(df[label_col])
        # Optional: Print the mapping
        print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    elif label_col in df.columns:
        print(f"Target variable '{label_col}' is already numerical.")

    print(f"Preprocessing complete. Final shape: {df.shape}")
    print(f"Shape change: {initial_shape} -> {df.shape}")
    return df


if __name__ == "__main__":
    # Define absolute paths for data directories
    # Update the base_path to match your project's root directory
    base_path = r"E:\intern\project\NIDS\NIDS_Project" # Use raw string for Windows path
    raw_data_path = os.path.join(base_path, "data", "raw")
    processed_data_path = os.path.join(base_path, "data", "processed")

    # Load the raw data
    try:
        raw_df = load_data(raw_data_path)
    except FileNotFoundError as e:
        print(e)
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        exit()

    # Explore the raw data
    explore_data(raw_df)

    # Apply basic preprocessing (using the memory-efficient version)
    processed_df = basic_preprocessing(raw_df.copy())

    # Optional: Save the preprocessed data for later use (saves time on future runs)
    processed_file_path = os.path.join(processed_data_path, "preprocessed_data.csv")
    os.makedirs(processed_data_path, exist_ok=True) # Create directory if it doesn't exist
    print(f"\nSaving preprocessed data to {processed_file_path}...")
    processed_df.to_csv(processed_file_path, index=False)
    print(f"Preprocessed data saved successfully to {processed_file_path}")

    # Display final info after preprocessing
    print("\n--- Final Dataset Info After Preprocessing ---")
    print(processed_df.info())
    print("\n--- Final Dataset Head ---")
    print(processed_df.head())
    print("\n--- Final Target Variable Distribution ---")
    if 'Label' in processed_df.columns:
        print(processed_df['Label'].value_counts())
    else:
        print("Target variable 'Label' not found in the final processed DataFrame.")
