"""
Data Cleaning Module for Titanic Analysis
Author: Alireza Barzin Zanganeh
Description: Utility functions for cleaning and preprocessing Titanic dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any


def load_titanic_data() -> pd.DataFrame:
    """
    Load the Titanic dataset from seaborn.
    
    Returns:
        pd.DataFrame: Raw Titanic dataset
    """
    import seaborn as sns
    df = sns.load_dataset('titanic')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check and report missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Missing value counts per column
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage
    })
    
    print("Missing Values Summary:")
    print("=" * 30)
    print(missing_summary[missing_summary['Missing Count'] > 0])
    
    return missing_values


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the Titanic dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    df_clean = df.copy()
    
    # Handle age missing values with median
    age_median = df_clean['age'].median()
    df_clean['age'] = df_clean['age'].fillna(age_median)
    print(f"✅ Age missing values filled with median: {age_median}")
    
    # Handle embarked missing values with mode
    embarked_mode = df_clean['embarked'].mode()[0]
    df_clean['embarked'] = df_clean['embarked'].fillna(embarked_mode)
    print(f"✅ Embarked missing values filled with mode: {embarked_mode}")
    
    # Drop deck column (too many missing values)
    if 'deck' in df_clean.columns:
        df_clean = df_clean.drop('deck', axis=1)
        print("✅ Deck column dropped (77% missing values)")
    
    # Drop rows with missing embark_town
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['embark_town'])
    rows_dropped = initial_rows - len(df_clean)
    if rows_dropped > 0:
        print(f"✅ Dropped {rows_dropped} rows with missing embark_town")
    
    return df_clean


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types for efficient analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with optimized data types
    """
    df_typed = df.copy()
    
    # Convert to categorical
    categorical_columns = ['sex', 'embarked', 'class', 'who', 'embark_town']
    for col in categorical_columns:
        if col in df_typed.columns:
            df_typed[col] = df_typed[col].astype('category')
    
    # Convert to boolean
    boolean_columns = ['adult_male', 'alone']
    for col in boolean_columns:
        if col in df_typed.columns:
            df_typed[col] = df_typed[col].astype('bool')
    
    # Convert to int
    int_columns = ['survived', 'pclass', 'sibsp', 'parch']
    for col in int_columns:
        if col in df_typed.columns:
            df_typed[col] = df_typed[col].astype('int')
    
    print("✅ Data types optimized for efficient analysis")
    return df_typed


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with duplicates removed
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    if duplicates_removed > 0:
        print(f"✅ Removed {duplicates_removed} duplicate rows")
    else:
        print("✅ No duplicate rows found")
    
    return df_clean


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, float, float]:
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to analyze
        
    Returns:
        Tuple[pd.DataFrame, float, float]: Outliers dataframe, lower bound, upper bound
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers, lower_bound, upper_bound


def cap_outliers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Cap outliers using the IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): List of column names to process
        
    Returns:
        pd.DataFrame: Dataframe with outliers capped
    """
    df_capped = df.copy()
    
    for column in columns:
        if column not in df_capped.columns:
            print(f"⚠️  Column '{column}' not found in dataframe")
            continue
            
        outliers, lower_bound, upper_bound = detect_outliers_iqr(df_capped, column)
        
        # Cap the outliers
        df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"✅ {column}: {len(outliers)} outliers capped (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
    
    return df_capped


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return summary statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, Any]: Data quality summary
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['category', 'object']).columns.tolist(),
        'boolean_columns': df.select_dtypes(include=['bool']).columns.tolist()
    }
    
    print("Data Quality Report:")
    print("=" * 25)
    print(f"Total Rows: {quality_report['total_rows']:,}")
    print(f"Total Columns: {quality_report['total_columns']}")
    print(f"Missing Values: {quality_report['missing_values']}")
    print(f"Duplicate Rows: {quality_report['duplicate_rows']}")
    print(f"Memory Usage: {quality_report['memory_usage']:,} bytes")
    print(f"Numeric Columns: {len(quality_report['numeric_columns'])}")
    print(f"Categorical Columns: {len(quality_report['categorical_columns'])}")
    print(f"Boolean Columns: {len(quality_report['boolean_columns'])}")
    
    return quality_report


def clean_titanic_dataset(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Complete data cleaning pipeline for Titanic dataset.
    
    Args:
        df (pd.DataFrame, optional): Input dataframe. If None, loads from seaborn.
        
    Returns:
        pd.DataFrame: Cleaned and processed dataframe
    """
    print("🧹 Starting Titanic Data Cleaning Pipeline")
    print("=" * 45)
    
    # Load data if not provided
    if df is None:
        df = load_titanic_data()
    
    # Step 1: Check initial data quality
    print("\n1. Initial Data Quality Check:")
    initial_quality = validate_data_quality(df)
    
    # Step 2: Handle missing values
    print("\n2. Handling Missing Values:")
    df_clean = handle_missing_values(df)
    
    # Step 3: Convert data types
    print("\n3. Converting Data Types:")
    df_clean = convert_data_types(df_clean)
    
    # Step 4: Remove duplicates
    print("\n4. Removing Duplicates:")
    df_clean = remove_duplicates(df_clean)
    
    # Step 5: Handle outliers
    print("\n5. Handling Outliers:")
    numeric_columns = ['age', 'fare']
    df_clean = cap_outliers(df_clean, numeric_columns)
    
    # Step 6: Final quality check
    print("\n6. Final Data Quality Check:")
    final_quality = validate_data_quality(df_clean)
    
    print("\n✅ Data cleaning pipeline completed successfully!")
    print(f"📊 Dataset shape: {df_clean.shape}")
    
    return df_clean


def export_cleaned_data(df: pd.DataFrame, filename: str = 'data/titanic_cleaned.csv') -> None:
    """
    Export cleaned dataset to CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        filename (str): Output filename
    """
    try:
        df.to_csv(filename, index=False)
        print(f"✅ Cleaned dataset exported to '{filename}'")
    except Exception as e:
        print(f"❌ Error exporting dataset: {e}")


if __name__ == "__main__":
    # Example usage
    print("Running Titanic Data Cleaning Pipeline...")
    
    # Clean the dataset
    cleaned_df = clean_titanic_dataset()
    
    # Export cleaned data
    export_cleaned_data(cleaned_df)
    
    print("\n🎉 Data cleaning completed successfully!")