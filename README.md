# Credit Loan Analysis and Modeling

This repository contains a comprehensive set of resources, including notebooks, data, and visualizations, designed to analyze and model credit loan data. The project leverages advanced data analysis techniques, machine learning models, and clustering methods to derive insights and build predictive models for credit risk assessment.

## Repository Contents

- **`Association_rules.ipynb`**: Implements association rule mining to uncover meaningful patterns and relationships in the credit loan data.
  
- **`EDA_credit_loan.ipynb`**: Conducts a thorough exploratory data analysis (EDA), providing insights into the structure and trends of the dataset through descriptive statistics and visualizations.

- **`credit_loan_classification.ipynb`**: Focuses on developing classification models, such as Decision Trees and Random Forest, to predict loan default risks. Includes model evaluation metrics like confusion matrix and classification reports.

- **`credit_loan_clustering_analysis.ipynb`**: Explores clustering techniques, including K-means and Hierarchical clustering, for segmenting data into distinct groups. Utilizes dimensionality reduction methods like PCA for effective visualization.

- **`credit_scoring.csv`**: The dataset used in this project, containing anonymized data related to credit scoring and financial behavior.

## Dataset Overview

The dataset includes key variables such as:
- **Default**: Indicates whether a client defaulted on a loan (Yes/No).
- **Prct_uso_tc**: Percentage of credit card usage in the last month.
- **Edad**: Client's age.
- **Mto_ingreso_mensual**: Monthly income amount.
- **Nro_prestamo_retrasados**: Number of loans with payment delays over 3 months in the past 3 years.
- And several other variables capturing financial and demographic information.

## Key Analyses and Insights

1. **Missing Data Handling**:
   - Missing values in `Mto_ingreso_mensual` were imputed using a predictive regression model.
   - Median imputation was applied for missing values in `Nro_dependiente`.

2. **Exploratory Data Analysis**:
   - Visualized the distribution of defaults using pie charts.
   - Highlighted trends and relationships between variables like debt-to-income ratio and age.

3. **Clustering**:
   - Used the Elbow Method and Silhouette Analysis to determine the optimal number of clusters for K-means.
   - Visualized clusters in a 2D PCA projection and created dendrograms for hierarchical clustering.

4. **Classification**:
   - Developed Decision Tree classifiers with hyperparameter tuning.
   - Evaluated model performance through cross-validation, analyzing accuracy vs. complexity parameters.

## Visualizations

- **Pie Chart**: Displays the proportion of clients who defaulted versus those who did not.
- **Elbow and Silhouette Plots**: Help determine the optimal number of clusters for K-means.
- **PCA Projection and Dendrogram**: Visualize clusters for both K-means and Hierarchical clustering methods.
- **Decision Tree Performance**: Plots of accuracy vs. tree complexity and depth to optimize the classification model.

## Prerequisites

To run the code, the following dependencies are required:
- Python 3.7+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `mlxtend`, etc.

Install all dependencies via:
```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repo>.git
   ```
2. Navigate to the project directory:
   ```bash
   cd <repository-folder>
   ```
3. Run the Jupyter notebooks in the recommended order:
   - Start with `EDA_credit_loan.ipynb` to understand the dataset.
   - Proceed to `credit_loan_classification.ipynb` for predictive modeling.
   - Use `credit_loan_clustering_analysis.ipynb` for clustering analyses.
   - Finally, explore `Association_rules.ipynb` for association rule mining.

## Conclusion

This project provides a complete pipeline for credit loan analysis, from data cleaning and visualization to predictive modeling and pattern discovery. The insights derived can aid in assessing credit risk, segmenting customers, and identifying key behaviors linked to defaults.

## Contributors

Developed by [FantacherJOY](https://github.com/FantacherJOY) and team for our group project.

## License

This project is licensed under the [MIT License](LICENSE).

