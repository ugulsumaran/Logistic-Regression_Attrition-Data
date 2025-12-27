# Employee Attrition Prediction with Logistic Regression
This project develops an advanced Logistic Regression model to predict employee attrition using IBM's fictional HR dataset. Far beyond a simple model, it includes comprehensive data preprocessing, feature engineering, multicollinearity management, and handling of class imbalance with advanced techniques.
The project provides a professional end-to-end data science pipeline example, addressing real-world complexities commonly encountered in predictive modeling.
## Project Description
High employee turnover imposes significant costs on companies. This project builds a predictive model to estimate the likelihood of an employee leaving based on various features. The approach goes beyond basic logistic regression by incorporating automated feature classification, rare category grouping, outlier analysis, correlation management, and meaningful new feature creation.
Given the imbalanced dataset (low attrition rate), class weights are applied, and evaluation focuses primarily on ROC-AUC.
## Dataset
- File:WA_Fn-UseC_-HR-Employee-Attrition.csv
- Source: IBM HR Analytics Employee Attrition & Performance (popular fictional dataset on Kaggle)
- Size: 1470 rows, 35 columns
- Target Variable:Attrition (Yes/No)
- Key Features: Age, MonthlyIncome, OverTime, JobSatisfaction, YearsAtCompany, DistanceFromHome, etc.
- The dataset is included in the repository.

## Libraries Used
- Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (LogisticRegression, StandardScaler, metrics, train_test_split, etc.)
- Custom helper functions (for feature classification, rare encoding, etc.)

## Project Steps (Detailed Overview)

- Data Loading and Exploratory Data Analysis (EDA):
  - Basic libraries such as Pandas, NumPy, Scikit-learn (LogisticRegression, train_test_split, StandardScaler, metrics), Matplotlib, and Seaborn are imported.   
  - Examining data structure (head/tail, shape(1470, 35), info)
  - Visualizing distributions of numerical and categorical variables with histograms
  - Analyzing target variable distribution (detecting class imbalance)

- Data Preprocessing:
  - Removing constant-value columns (EmployeeCount, Over18, StandardHours)
  - Automatic column classification: categorical, cardinal, numerical, numerical-but-categorical
  - Missing value check (none present)
  - Outlier detection in numerical features
  - Identifying and grouping rare categories as "Rare"

- Correlation Analysis:
  - Heatmap of correlations among encoded features
  - Detecting highly correlated pairs (≥ 0.7 threshold)
    - Correlation matrix: Strong relationships identified, for example: JobLevel and MonthlyIncome: +0.95 correlation
    - Negative correlations between departments (Sales vs R&D: -0.91)
  - Ranking correlations with the target variable (identifying most influential features)

- Feature Engineering:
  - Creating meaningful new features to reduce multicollinearity and capture deeper relationships: IncomePerLevel, ExperiencePerLevel, IncomePerExperienceYear
RoleManagerStability, ManagerTenureRatio, PromotionLag
OvertimeBurden, CommuteStress, OverallSatisfaction
- As a result, the dataset is reduced to (1470, 33) columns.

  - Dropping redundant original features

- Model Preparation:
  - One-Hot Encoding for remaining categorical variables
  - Stratified train/test split (70% train, 30% test)
  - Scaling numerical features with StandardScaler


- Logistic Regression Model:
  - L2 regularization
  - class_weight='balanced' to handle class imbalance
  - Model training and feature importance analysis via coefficients

- Evaluation:
  - ROC-AUC score (primary metric)
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix
  - Interpretation of key attrition risk factors (e.g., overtime burden, promotion delays)
    - Key features increasing attrition probability (positive coefficients):
      - JobRole_Sales Executive: +0.908 (strongest effect) OverTimeNum (Overtime): +0.741
    - Key features decreasing attrition probability (negative coefficients):
      -OverallSatisfaction: -0.714 (strongest protective effect) IncomePerLevel: -0.381


## How to Run

- Clone the repository: 
 ```bash
git clone https://github.com/ugulsumaran/LogisticRegression_Attrition.git
```
- Open the Jupyter Notebook:
Attrition_LogisticRegression.ipynb → Version with outputs (to view results)
Attrition_LogisticRegression_clear.ipynb → Clean code version

- Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
- Run the notebook cells sequentially.

## Results
Thanks to engineered features, the model achieves meaningful performance. The strongest predictors of attrition are typically overtime burden, promotion delays, income-experience imbalances, and overall satisfaction scores.
This project is much more than a basic model: it is a comprehensive pipeline addressing realistic data challenges (outliers, rare categories, multicollinearity, imbalance).
<img width="874" height="399" alt="image" src="https://github.com/user-attachments/assets/72f5d17b-77d9-456f-b03e-d0199b19b259" />
