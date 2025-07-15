# Titanic Survival Analysis: Patterns and Insights

A comprehensive data analysis project examining survival patterns from the 1912 Titanic disaster using Python, pandas, and advanced visualization techniques.

## Project Overview

This project analyzes the Titanic dataset to uncover factors that influenced passenger survival during the historic disaster. Through exploratory data analysis, data cleaning, feature engineering, and statistical visualization, we investigate how demographic and socio-economic factors affected survival rates.

### Key Research Questions
- How did gender, class, and age influence survival rates?
- What evidence supports the "women and children first" evacuation policy?
- Which passenger characteristics were most predictive of survival?
- How did fare prices correlate with survival outcomes?

## Key Findings

### Survival Statistics
- **Overall Survival Rate**: 38.38% (342 out of 891 passengers)
- **Gender Impact**: Women had 74.2% survival vs. 18.9% for men
- **Class Disparity**: First-class (63.0%) vs. Third-class (24.2%) survival
- **Age Factor**: Children had 59.1% survival rate vs. 37% for adults

### Critical Insights
1. **"Women and Children First" Policy**: Clear evidence of implementation with women having 4x higher survival rates
2. **Social Class Impact**: First-class passengers had 2.6x better survival odds than third-class
3. **Fare Correlation**: Positive correlation (0.25) between fare paid and survival likelihood
4. **Family Size Effect**: Small families had better survival rates than large families or solo travelers

## Technical Implementation

### Technologies Used
- **Python 3.8+** - Core programming language
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Seaborn** - Statistical data visualization
- **Matplotlib** - Plotting and visualization
- **Jupyter Notebook** - Interactive development environment

### Data Processing Pipeline
1. **Data Loading**: Imported Titanic dataset from seaborn
2. **Data Cleaning**: Handled missing values, removed duplicates, optimized data types
3. **Outlier Treatment**: Applied IQR method for fare and age outliers
4. **Feature Engineering**: Created age groups, family size, and fare brackets
5. **Statistical Analysis**: Computed survival rates across multiple dimensions
6. **Visualization**: Generated 15+ charts including heatmaps, violin plots, and facet grids

## Project Structure

```
titanic-survival-analysis/
├── README.md                 # This file
├── titanic_analysis.ipynb    # Complete Jupyter notebook
├── data/
│   └── titanic_processed.csv # Cleaned dataset
├── visualizations/
│   ├── age_distribution.png
│   ├── survival_by_class_gender.png
│   ├── correlation_heatmap.png
│   └── fare_violin_plots.png
├── src/
│   ├── data_cleaning.py      # Data preprocessing functions
│   ├── feature_engineering.py # Feature creation utilities
│   └── visualization.py      # Plotting functions
└── requirements.txt          # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git for version control

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/titanic-survival-analysis.git
cd titanic-survival-analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook titanic_analysis.ipynb
```

## Analysis Highlights

### Data Quality & Preprocessing
- Handled 177 missing age values using median imputation
- Removed 688 missing deck values (77% missing)
- Converted categorical variables for efficient analysis
- Applied outlier capping using IQR method

### Feature Engineering
- **Age Groups**: Child (0-17), Adult (18-64), Senior (65+)
- **Family Size**: Total family members aboard
- **Fare Brackets**: Quartile-based fare categorization
- **Isolation Index**: Solo traveler identification

### Statistical Insights
- **Correlation Analysis**: Fare shows strongest positive correlation with survival (0.25)
- **Cross-tabulation**: Detailed survival breakdown by multiple variables
- **Distribution Analysis**: Age and fare distributions reveal passenger demographics

## Key Visualizations

1. **Survival Rate by Class and Gender** - Bar chart showing intersectional survival patterns
2. **Age Distribution Analysis** - Histogram with survival overlay
3. **Correlation Heatmap** - Numerical variable relationships
4. **Fare vs. Class Violin Plots** - Distribution comparison by survival status
5. **Family Size Impact** - Count plot with survival rates

## Methodology

### Data Cleaning Process
```python
# Missing value strategy
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df.drop('deck', axis=1, inplace=True)  # 77% missing
```

### Feature Engineering Examples
```python
# Age categorization
df['age_group'] = pd.cut(df['age'], bins=[0, 17, 64, 130], 
                        labels=['Child', 'Adult', 'Senior'])

# Family size calculation
df['family_size'] = df['sibsp'] + df['parch'] + 1
```

## Future Enhancements

- **Machine Learning Models**: Implement predictive modeling (Random Forest, Logistic Regression)
- **Interactive Dashboard**: Create Streamlit/Dash application
- **Additional Datasets**: Integrate crew survival data
- **Temporal Analysis**: Time-series analysis of evacuation patterns
- **Geographic Analysis**: Departure port survival patterns

## Business Applications

This analysis methodology can be applied to:
- Emergency response planning
- Risk assessment modeling
- Demographic trend analysis
- Social policy impact evaluation
- Insurance actuarial modeling

## Contact & Connect

**Alireza Barzin Zanganeh**
- 📧 Email: abarzinzanganeh@gmail.com
- 💼 LinkedIn: [linkedin.com/in/alireza-barzin-zanganeh-2a9909126](https://linkedin.com/in/alireza-barzin-zanganeh-2a9909126)
- 📍 Location: WA, USA

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: Seaborn library's built-in Titanic dataset
- Inspiration: Kaggle Titanic Machine Learning competition
- Historical context: Encyclopedia Titanica and maritime history resources

