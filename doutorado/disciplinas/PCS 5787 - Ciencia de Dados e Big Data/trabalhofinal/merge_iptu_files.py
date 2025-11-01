#!/usr/bin/env python3
"""
Script to merge all IPTU_*.csv files from the data folder.
Each file's year is extracted from the filename and added as a 'year' column.
"""

import pandas as pd
import glob
import os
from pathlib import Path

def merge_iptu_files(data_folder="data", output_file="IPTU_merged_all_years.csv"):
    """
    Merge all IPTU_*.csv files from the specified data folder.
    
    Args:
        data_folder (str): Path to the folder containing IPTU CSV files
        output_file (str): Name of the output merged file
    """
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    data_path = script_dir / data_folder
    
    # Find all IPTU_*.csv files
    pattern = str(data_path / "IPTU_*.csv")
    iptu_files = glob.glob(pattern)
    
    # Filter out any existing merged files to avoid including them
    iptu_files = [f for f in iptu_files if not f.endswith("_merged.csv") and not f.endswith("_merged_all_years.csv")]
    
    if not iptu_files:
        print("No IPTU_*.csv files found in the data folder.")
        return
    
    print(f"Found {len(iptu_files)} IPTU files to merge:")
    for file in iptu_files:
        print(f"  - {os.path.basename(file)}")
    
    # List to store dataframes
    dataframes = []
    
    # Process each file
    for file_path in iptu_files:
        try:
            # Extract year from filename (last 4 characters before .csv)
            filename = os.path.basename(file_path)
            year = filename.split('_')[1].split('.')[0]  # Extract year from IPTU_YYYY.csv
            
            print(f"Processing {filename} (Year: {year})...")
            
            # Read the CSV file
            df = pd.read_csv(file_path, sep=';', encoding='utf-8')
            
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
    
    # Merge all dataframes
    print("\nMerging all dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Sort by year and other relevant columns for better organization
    merged_df = merged_df.sort_values(['year', 'NUMERO DO CONTRIBUINTE'])
    
    # Save the merged file
    output_path = data_path / output_file
    merged_df.to_csv(output_path, sep=';', index=False, encoding='utf-8')
    
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
    merge_iptu_files()
