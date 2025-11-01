#!/usr/bin/env python3
"""
Script to merge all ITBI_*.csv files from the data folder.
Each file's year is extracted from the filename and added as a 'year' column.
"""

import pandas as pd
import glob
import os
from pathlib import Path

def merge_itbi_files(data_folder="data", output_file="ITBI_merged_all_years.csv"):
    """
    Merge all ITBI_*.csv files from the specified data folder.
    
    Args:
        data_folder (str): Path to the folder containing ITBI CSV files
        output_file (str): Name of the output merged file
    """
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    data_path = script_dir / data_folder
    
    # Find all ITBI_*.csv files
    pattern = str(data_path / "ITBI_*.csv")
    itbi_files = glob.glob(pattern)
    
    # Filter out any existing merged files to avoid including them
    itbi_files = [f for f in itbi_files if not f.endswith("_merged.csv") and not f.endswith("_merged_all_years.csv")]
    
    if not itbi_files:
        print("No ITBI_*.csv files found in the data folder.")
        return
    
    print(f"Found {len(itbi_files)} ITBI files to merge:")
    for file in itbi_files:
        print(f"  - {os.path.basename(file)}")
    
    # List to store dataframes
    dataframes = []
    
    # Process each file
    for file_path in itbi_files:
        try:
            # Extract year from filename (last 4 characters before .csv)
            filename = os.path.basename(file_path)
            year = filename.split('_')[1].split('.')[0]  # Extract year from ITBI_YYYY.csv
            
            print(f"Processing {filename} (Year: {year})...")
            
            # Try different encodings to handle the special characters
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, sep=',', encoding=encoding)
                    print(f"  - Successfully read with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"  - Error with {encoding}: {str(e)}")
                    continue
            
            if df is None:
                print(f"  - Could not read file with any encoding")
                continue
            
            # Add year column
            df['year'] = int(year)
            
            # Add to list
            dataframes.append(df)
            
            print(f"  - Loaded {len(df)} rows")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not dataframes:
        print("No files were successfully processed.")
        return
    
    # Check if all dataframes have the same columns
    print("\nChecking column consistency...")
    first_columns = set(dataframes[0].columns)
    for i, df in enumerate(dataframes[1:], 1):
        current_columns = set(df.columns)
        if first_columns != current_columns:
            print(f"Warning: DataFrame {i+1} has different columns than the first one.")
            print(f"First dataframe columns: {sorted(first_columns)}")
            print(f"DataFrame {i+1} columns: {sorted(current_columns)}")
            print(f"Common columns: {sorted(first_columns.intersection(current_columns))}")
            print(f"Missing in DataFrame {i+1}: {sorted(first_columns - current_columns)}")
            print(f"Extra in DataFrame {i+1}: {sorted(current_columns - first_columns)}")
    
    # Merge all dataframes
    print("\nMerging all dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Sort by year and other relevant columns for better organization
    if 'Data de Transação' in merged_df.columns:
        merged_df = merged_df.sort_values(['year', 'Data de Transação'])
    else:
        merged_df = merged_df.sort_values(['year'])
    
    # Save the merged file
    output_path = data_path / output_file
    merged_df.to_csv(output_path, sep=',', index=False, encoding='utf-8')
    
    print(f"\nMerged file saved as: {output_path}")
    print(f"Total rows in merged file: {len(merged_df)}")
    print(f"Years included: {sorted(merged_df['year'].unique())}")
    print(f"Columns in merged file: {list(merged_df.columns)}")
    
    # Display basic statistics
    print(f"\nBasic statistics:")
    print(f"  - Total records: {len(merged_df)}")
    print(f"  - Records per year:")
    year_counts = merged_df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"    {year}: {count:,} records")

if __name__ == "__main__":
    # Run the merge function
    merge_itbi_files()
