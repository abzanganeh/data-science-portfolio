"""
Feature Engineering Module for Titanic Analysis
Author: Alireza Barzin Zanganeh
Description: Functions for creating new features from Titanic dataset
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re


def create_age_groups(df: pd.DataFrame, 
                     age_column: str = 'age',
                     bins: List[float] = [0, 17, 64, 130],
                     labels: List[str] = ['Child', 'Adult', 'Senior']) -> pd.DataFrame:
    """
    Create age group categories from continuous age data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        age_column (str): Name of the age column
        bins (List[float]): Age bin boundaries
        labels (List[str]): Labels for age groups
        
    Returns:
        pd.DataFrame: Dataframe with age_group column added
    """
    df_feature = df.copy()
    
    df_feature['age_group'] = pd.cut(df_feature[age_column], 
                                   bins=bins, 
                                   labels=labels, 
                                   right=False)
    
    print(f"✅ Age groups created: {df_feature['age_group'].value_counts().to_dict()}")
    return df_feature


def create_family_features(df: pd.DataFrame,
                          sibsp_column: str = 'sibsp',
                          parch_column: str = 'parch') -> pd.DataFrame:
    """
    Create family-related features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        sibsp_column (str): Siblings/spouses column name
        parch_column (str): Parents/children column name
        
    Returns:
        pd.DataFrame: Dataframe with family features added
    """
    df_feature = df.copy()
    
    # Total family size (including passenger)
    df_feature['family_size'] = df_feature[sibsp_column] + df_feature[parch_column] + 1
    
    # Family size categories
    df_feature['family_size_category'] = pd.cut(df_feature['family_size'],
                                               bins=[0, 1, 4, 11],
                                               labels=['Alone', 'Small', 'Large'],
                                               right=False)
    
    # Is alone indicator
    df_feature['is_alone'] = (df_feature['family_size'] == 1).astype(int)
    
    # Has siblings/spouses
    df_feature['has_siblings_spouses'] = (df_feature[sibsp_column] > 0).astype(int)
    
    # Has parents/children
    df_feature['has_parents_children'] = (df_feature[parch_column] > 0).astype(int)
    
    print("✅ Family features created:")
    print(f"   - Family size distribution: {df_feature['family_size'].value_counts().sort_index().to_dict()}")
    print(f"   - Family categories: {df_feature['family_size_category'].value_counts().to_dict()}")
    print(f"   - Alone passengers: {df_feature['is_alone'].sum()}")
    
    return df_feature


def create_fare_features(df: pd.DataFrame,
                        fare_column: str = 'fare') -> pd.DataFrame:
    """
    Create fare-related features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        fare_column (str): Fare column name
        
    Returns:
        pd.DataFrame: Dataframe with fare features added
    """
    df_feature = df.copy()
    
    # Fare brackets using quantiles
    df_feature['fare_bracket'] = pd.qcut(df_feature[fare_column],
                                       q=4,
                                       labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Fare per person (fare divided by family size)
    df_feature['fare_per_person'] = df_feature[fare_column] / df_feature['family_size']
    
    # Fare categories based on statistical distribution
    fare_mean = df_feature[fare_column].mean()
    fare_std = df_feature[fare_column].std()
    
    def categorize_fare(fare):
        if fare < fare_mean - fare_std:
            return 'Below Average'
        elif fare > fare_mean + fare_std:
            return 'Above Average'
        else:
            return 'Average'
    
    df_feature['fare_category'] = df_feature[fare_column].apply(categorize_fare)
    
    print("✅ Fare features created:")
    print(f"   - Fare brackets: {df_feature['fare_bracket'].value_counts().to_dict()}")
    print(f"   - Fare categories: {df_feature['fare_category'].value_counts().to_dict()}")
    
    return df_feature


def extract_title_from_name(df: pd.DataFrame,
                           name_column: str = 'name') -> pd.DataFrame:
    """
    Extract title from passenger names.
    
    Args:
        df (pd.DataFrame): Input dataframe
        name_column (str): Name column name
        
    Returns:
        pd.DataFrame: Dataframe with title column added
    """
    df_feature = df.copy()
    
    # Extract title using regex
    df_feature['title'] = df_feature[name_column].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_counts = df_feature['title'].value_counts()
    rare_titles = title_counts[title_counts < 10].index.tolist()
    
    df_feature['title'] = df_feature['title'].replace(rare_titles, 'Rare')
    
    # Create title categories
    title_mapping = {
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master',
        'Dr': 'Officer',
        'Rev': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        'Capt': 'Officer',
        'Rare': 'Rare'
    }
    
    df_feature['title_category'] = df_feature['title'].map(title_mapping)
    
    print("✅ Title features created:")
    print(f"   - Titles: {df_feature['title'].value_counts().to_dict()}")
    print(f"   - Title categories: {df_feature['title_category'].value_counts().to_dict()}")
    
    return df_feature


def create_deck_features(df: pd.DataFrame,
                        cabin_column: str = 'cabin') -> pd.DataFrame:
    """
    Create deck-related features from cabin information.
    
    Args:
        df (pd.DataFrame): Input dataframe
        cabin_column (str): Cabin column name
        
    Returns:
        pd.DataFrame: Dataframe with deck features added
    """
    df_feature = df.copy()
    
    # Extract deck letter
    df_feature['deck'] = df_feature[cabin_column].str.extract(r'([A-Za-z])', expand=False)
    
    # Has cabin information
    df_feature['has_cabin'] = (~df_feature[cabin_column].isna()).astype(int)
    
    # Cabin number (if available)
    df_feature['cabin_number'] = df_feature[cabin_column].str.extract(r'(\d+)', expand=False)
    df_feature['cabin_number'] = pd.to_numeric(df_feature['cabin_number'], errors='coerce')
    
    # Multiple cabins indicator
    df_feature['multiple_cabins'] = (df_feature[cabin_column].str.count(' ') > 0).astype(int)
    
    print("✅ Deck features created:")
    if 'deck' in df_feature.columns:
        print(f"   - Deck distribution: {df_feature['deck'].value_counts().to_dict()}")
    print(f"   - Has cabin: {df_feature['has_cabin'].sum()} passengers")
    print(f"   - Multiple cabins: {df_feature['multiple_cabins'].sum()} passengers")
    
    return df_feature


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between existing variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with interaction features added
    """
    df_feature = df.copy()
    
    # Gender-Class interaction
    df_feature['gender_class'] = df_feature['sex'].astype(str) + '_' + df_feature['class'].astype(str)
    
    # Age-Class interaction
    df_feature['age_class'] = df_feature['age_group'].astype(str) + '_' + df_feature['class'].astype(str)
    
    # Family-Class interaction
    df_feature['family_class'] = df_feature['family_size_category'].astype(str) + '_' + df_feature['class'].astype(str)
    
    # Embarked-Class interaction
    df_feature['embarked_class'] = df_feature['embarked'].astype(str) + '_' + df_feature['class'].astype(str)
    
    print("✅ Interaction features created:")
    print(f"   - Gender-Class combinations: {df_feature['gender_class'].nunique()}")
    print(f"   - Age-Class combinations: {df_feature['age_class'].nunique()}")
    print(f"   - Family-Class combinations: {df_feature['family_class'].nunique()}")
    
    return df_feature


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistical features based on groupings.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with statistical features added
    """
    df_feature = df.copy()
    
    # Average fare by class
    class_fare_mean = df_feature.groupby('class')['fare'].transform('mean')
    df_feature['fare_vs_class_mean'] = df_feature['fare'] / class_fare_mean
    
    # Average age by class
    class_age_mean = df_feature.groupby('class')['age'].transform('mean')
    df_feature['age_vs_class_mean'] = df_feature['age'] / class_age_mean
    
    # Passenger count by embarkation port
    df_feature['embarked_count'] = df_feature.groupby('embarked')['embarked'].transform('count')
    
    # Family size rank within class
    df_feature['family_size_rank'] = df_feature.groupby('class')['family_size'].rank(method='dense')
    
    print("✅ Statistical features created:")
    print(f"   - Fare vs class mean: {df_feature['fare_vs_class_mean'].describe()}")
    print(f"   - Age vs class mean: {df_feature['age_vs_class_mean'].describe()}")
    
    return df_feature


def encode_categorical_features(df: pd.DataFrame,
                              columns_to_encode: List[str] = None,
                              encoding_method: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical features for machine learning.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns_to_encode (List[str]): List of columns to encode
        encoding_method (str): 'onehot' or 'label'
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    df_encoded = df.copy()
    
    if columns_to_encode is None:
        columns_to_encode = ['sex', 'embarked', 'class', 'who', 'age_group', 
                           'family_size_category', 'fare_bracket', 'title_category']
    
    # Filter columns that exist in the dataframe
    columns_to_encode = [col for col in columns_to_encode if col in df_encoded.columns]
    
    if encoding_method == 'onehot':
        # One-hot encoding
        df_encoded = pd.get_dummies(df_encoded, columns=columns_to_encode, 
                                  prefix=columns_to_encode, drop_first=True)
        print(f"✅ One-hot encoded {len(columns_to_encode)} categorical columns")
        
    elif encoding_method == 'label':
        # Label encoding
        from sklearn.preprocessing import LabelEncoder
        
        for col in columns_to_encode:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
        
        print(f"✅ Label encoded {len(columns_to_encode)} categorical columns")
    
    return df_encoded


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features using the complete feature engineering pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with all engineered features
    """
    print("🔧 Starting Feature Engineering Pipeline")
    print("=" * 40)
    
    # Step 1: Age groups
    print("\n1. Creating age groups...")
    df_features = create_age_groups(df)
    
    # Step 2: Family features
    print("\n2. Creating family features...")
    df_features = create_family_features(df_features)
    
    # Step 3: Fare features
    print("\n3. Creating fare features...")
    df_features = create_fare_features(df_features)
    
    # Step 4: Title features
    print("\n4. Extracting title features...")
    df_features = extract_title_from_name(df_features)
    
    # Step 5: Deck features (if cabin column exists)
    if 'cabin' in df_features.columns:
        print("\n5. Creating deck features...")
        df_features = create_deck_features(df_features)
    
    # Step 6: Interaction features
    print("\n6. Creating interaction features...")
    df_features = create_interaction_features(df_features)
    
    # Step 7: Statistical features
    print("\n7. Creating statistical features...")
    df_features = create_statistical_features(df_features)
    
    print(f"\n✅ Feature engineering completed!")
    print(f"📊 Final dataset shape: {df_features.shape}")
    print(f"📈 Features created: {df_features.shape[1] - df.shape[1]} new features")
    
    return df_features


def get_feature_importance_summary(df: pd.DataFrame, target_column: str = 'survived') -> pd.DataFrame:
    """
    Generate feature importance summary for created features.
    
    Args:
        df (pd.DataFrame): Input dataframe with features
        target_column (str): Target variable column name
        
    Returns:
        pd.DataFrame: Feature importance summary
    """
    from scipy.stats import chi2_contingency
    import warnings
    warnings.filterwarnings('ignore')
    
    # Select numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Remove target column
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    feature_importance = []
    
    # Correlation for numeric features
    for col in numeric_cols:
        if col != target_column:
            try:
                correlation = df[col].corr(df[target_column])
                feature_importance.append({
                    'feature': col,
                    'type': 'numeric',
                    'importance': abs(correlation),
                    'metric': 'correlation'
                })
            except:
                pass
    
    # Chi-square for categorical features
    for col in categorical_cols:
        if col != target_column and col in df.columns:
            try:
                crosstab = pd.crosstab(df[col], df[target_column])
                chi2, p_value, dof, expected = chi2_contingency(crosstab)
                feature_importance.append({
                    'feature': col,
                    'type': 'categorical',
                    'importance': chi2,
                    'metric': 'chi2'
                })
            except:
                pass
    
    # Convert to DataFrame and sort
    importance_df = pd.DataFrame(feature_importance)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("🎯 Feature Importance Summary (Top 15):")
    print("=" * 40)
    print(importance_df.head(15))
    
    return importance_df


if __name__ == "__main__":
    # Example usage
    import seaborn as sns
    
    print("Running Feature Engineering Pipeline...")
    
    # Load sample data
    df = sns.load_dataset('titanic')
    
    # Create all features
    df_with_features = create_all_features(df)
    
    # Get feature importance
    feature_importance = get_feature_importance_summary(df_with_features)
    
    print("\n🎉 Feature engineering completed successfully!")
    print(f"Original features: {df.shape[1]}")
    print(f"Final features: {df_with_features.shape[1]}")
    print(f"New features created: {df_with_features.shape[1] - df.shape[1]}")
