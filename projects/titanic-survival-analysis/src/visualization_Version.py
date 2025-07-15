"""
Visualization Module for Titanic Analysis
Author: Alireza Barzin Zanganeh
Description: Functions for generating key plots and statistical visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

sns.set_palette("husl")
plt.style.use('seaborn-v0_8')


def plot_age_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot and save age distribution histogram with mean and median lines.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df['age'], bins=30, kde=True, color='skyblue', alpha=0.7)
    plt.title('Age Distribution of Titanic Passengers', fontsize=14, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["age"].mean():.1f}')
    plt.axvline(df['age'].median(), color='orange', linestyle=':', linewidth=2,
                label=f'Median: {df["age"].median():.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_survival_by_class_gender(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Bar chart of survival rates by class and gender.
    """
    survival_by_class_gender = df.groupby(['class', 'sex'])['survived'].mean().unstack()
    ax = survival_by_class_gender.plot(
        kind='bar',
        color=['lightcoral', 'lightblue'],
        figsize=(8, 6),
        width=0.8
    )
    plt.title('Survival Rate by Class and Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Passenger Class')
    plt.ylabel('Survival Rate')
    plt.legend(title='Gender')
    plt.xticks(rotation=0)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot correlation heatmap of numerical features.
    """
    numerical_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
    corr = df[numerical_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Numerical Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_fare_violin_plots(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Violin plots comparing fare distribution by class and survival.
    """
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='class', y='fare', hue='survived', data=df, split=True)
    plt.title('Fare Distribution by Class and Survival', fontsize=14, fontweight='bold')
    plt.ylabel('Fare (£)')
    plt.legend(title='Survived')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_family_size_impact(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Count plot of family size with survival rate overlay.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='family_size', hue='survived', data=df)
    plt.title('Family Size Impact on Survival', fontsize=14, fontweight='bold')
    plt.xlabel('Family Size')
    plt.ylabel('Passenger Count')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, color='black')
    plt.legend(title='Survived')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_age_survival_histogram(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Age histogram with survival overlay.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='age', hue='survived', multiple='stack', bins=30, alpha=0.8)
    plt.title('Age Distribution by Survival', fontsize=14, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_all_visualizations(df: pd.DataFrame, out_dir: str = "visualizations/") -> None:
    """
    Generate and save all key visualizations to the output directory.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving visualizations to {out_dir}")
    plot_age_distribution(df, save_path=os.path.join(out_dir, "age_distribution.png"))
    plot_survival_by_class_gender(df, save_path=os.path.join(out_dir, "survival_by_class_gender.png"))
    plot_correlation_heatmap(df, save_path=os.path.join(out_dir, "correlation_heatmap.png"))
    plot_fare_violin_plots(df, save_path=os.path.join(out_dir, "fare_violin_plots.png"))
    plot_family_size_impact(df, save_path=os.path.join(out_dir, "family_size_impact.png"))
    plot_age_survival_histogram(df, save_path=os.path.join(out_dir, "age_survival_histogram.png"))
    print("All visualizations saved.")


if __name__ == "__main__":
    import seaborn as sns
    print("Generating example visualizations...")

    # Load and process data
    df = sns.load_dataset('titanic')
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df = df.drop('deck', axis=1)
    df.dropna(subset=['embark_town'], inplace=True)
    df['family_size'] = df['sibsp'] + df['parch'] + 1

    save_all_visualizations(df)
