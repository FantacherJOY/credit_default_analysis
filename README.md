# Credit Loan Analysis and Modeling

This repository contains a comprehensive set of resources, including notebooks, data, and visualizations, designed to analyze and model credit loan data. The project leverages advanced data analysis techniques, machine learning models, clustering methods, and association rule mining to derive insights and build predictive models for credit risk assessment.

## Repository Contents

- **[`EDA_credit_loan.ipynb`](https://github.com/FantacherJOY/credit_default_analysis/blob/main/EDA_credit_loan.ipynb)**: Conducts a thorough exploratory data analysis (EDA), providing insights into the structure and trends of the dataset through descriptive statistics and visualizations.

- **[`Association_rules.ipynb`](https://github.com/FantacherJOY/credit_default_analysis/blob/main/Association_rules.ipynb)**: Implements association rule mining to uncover meaningful patterns and relationships in the credit loan data.

- **[`credit_loan_classification.ipynb`](https://github.com/FantacherJOY/credit_default_analysis/blob/main/credit_loan_classification.ipynb)**: Focuses on developing classification models, such as Random Forest, Naive Bayes, K-Nearest Neighbors, and Support Vector Machine (SVM), to predict loan default risks. Includes model evaluation metrics like confusion matrix and classification reports.

- **[`credit_loan_clusterring_analysis.ipynb`](https://github.com/FantacherJOY/credit_default_analysis/blob/main/credit_loan_clusterring_analysis.ipynb)**: Explores clustering techniques, including K-means and Hierarchical clustering, for segmenting data into distinct groups. Utilizes dimensionality reduction methods like PCA for effective visualization.

- **`credit_scoring.csv`**: The dataset used in this project, containing anonymized data related to credit scoring and financial behavior.

## Dataset Overview

The dataset includes key variables such as:

- **Default**: Client with more than 90 days without paying their loan (Y/N).
- **Prct_uso_tc**: Percentage of credit card usage in the last month (percentage).
- **Edad**: Age (integer).
- **Nro_prestamo_retrasados**: Number of loans with payment delays of more than 3 months in the last 3 years (integer).
- **Prct_deuda_vs_ingresos**: Financial debt-to-income ratio (percentage).
- **Mto_ingreso_mensual**: Monthly income amount (real).
- **Nro_prod_financieros_deuda**: Number of loans (including vehicle or mortgage loans) and number of credit cards last year (integer).
- **Nro_retraso_60dias**: Number of times the client has been over 60 days late in the last 3 years (integer).
- **Nro_creditos_hipotecarios**: Number of mortgage loans (integer).
- **Nro_retraso_ultm3anios**: Number of payment delays over 30 days in the last 3 years (integer).
- **Nro_dependiente**: Number of dependents (integer).

The dataset can be accessed from Kaggle: [Credit Default Dataset](https://www.kaggle.com/datasets/hugoferquiroz/credit-default-only-numbers).

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
   - Implemented several classifiers:
     - **Random Forest**: Achieved an accuracy of 93.46% with strong recall and precision for non-default predictions.
     - **Naive Bayes**: Provided an accuracy of 93% with adjusted parameters for optimal performance.
     - **Support Vector Machine (SVM)**: Demonstrated 94% accuracy with a linear kernel.
     - **K-Nearest Neighbors (KNN)**: Reached an accuracy of 93% using 2 neighbors.
   - Evaluated models using precision, recall, F1-score, and accuracy metrics.

5. **Association Rule Mining**:
   - Binned continuous variables such as monthly income and age into categories (e.g., "Very Low," "Low," "Medium," "High," "Very High" for income).
   - Derived rules to identify patterns between income groups and default status.
   - Visualized association rules using scatter plots of support, confidence, and lift metrics.
   - Example: Default rates by "Low" income group were analyzed, providing insights into high-risk categories.

## Visualizations

- **Pie Chart**: Displays the proportion of clients who defaulted versus those who did not.
- **Elbow and Silhouette Plots**: Help determine the optimal number of clusters for K-means.
- **PCA Projection and Dendrogram**: Visualize clusters for both K-means and Hierarchical clustering methods.
- **Decision Tree Performance**: Plots of accuracy vs. tree complexity and depth to optimize the classification model.
- **Scatter Plots of Association Rules**: Visualize rules by monthly income groups and age groups, showing support, confidence, and lift.

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
   git clone https://github.com/FantacherJOY/credit_default_analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit_default_analysis
   ```
3. Run the Jupyter notebooks in the recommended order:
   - Start with [`EDA_credit_loan.ipynb`](https://github.com/FantacherJOY/credit_default_analysis/blob/main/EDA_credit_loan.ipynb) to understand the dataset.
   - Proceed to [`credit_loan_classification.ipynb`](https://github.com/FantacherJOY/credit_default_analysis/blob/main/credit_loan_classification.ipynb) for predictive modeling.
   - Use [`credit_loan_clusterring_analysis.ipynb`](https://github.com/FantacherJOY/credit_default_analysis/blob/main/credit_loan_clusterring_analysis.ipynb) for clustering analyses.
   - Finally, explore [`Association_rules.ipynb`](https://github.com/FantacherJOY/credit_default_analysis/blob/main/Association_rules.ipynb) for association rule mining.

## Conclusion

This project provides a complete pipeline for credit loan analysis, from data cleaning and visualization to predictive modeling, clustering, and association rule discovery. The insights derived can aid in assessing credit risk, segmenting customers, and identifying key behaviors linked to defaults.

## Contributors

Developed by [FantacherJOY](https://github.com/FantacherJOY), [Mushaer Ahmed](https://github.com/mushaerahmed), and [Md MUzahid Khan](https://github.com/Khan10061646) for our group project.

## License

This project is licensed under the [MIT License](LICENSE).
