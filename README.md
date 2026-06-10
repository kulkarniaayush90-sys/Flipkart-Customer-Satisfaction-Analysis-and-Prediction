# Flipkart Customer Satisfaction Analysis and Prediction

## Project Overview

This project analyzes Flipkart customer support interactions to understand the key factors influencing customer satisfaction and to build machine learning models that can help identify dissatisfied customers.

Completed as part of the **Labmentix Internship Capstone Project**, this work combines exploratory data analysis, feature engineering, preprocessing, model comparison, class imbalance handling, target formulation evaluation, and production deployment recommendations.

The final recommendation is to use a **Random Forest model** with a **Low vs Not-Low** target formulation to proactively identify customers who are at risk of low satisfaction.

## Business Problem

Customer satisfaction is a critical metric for Flipkart because support interactions directly influence customer trust, repeat purchases, brand perception, and long-term retention.

In large-scale e-commerce operations, dissatisfied customers may arise from delayed responses, unresolved issues, product-category problems, channel inefficiencies, or inconsistent support quality. Predicting dissatisfaction early allows the business to:

- Prioritize high-risk customer interactions
- Improve service recovery workflows
- Reduce repeat complaints and escalations
- Support agents with better decision-making tools
- Improve overall customer experience

The goal of this project is not to automate customer service decisions, but to build a data-driven decision-support framework for identifying potential low-CSAT cases.

## Project Objectives

The main objectives of this project are:

- Perform Exploratory Data Analysis (EDA)
- Understand drivers of customer satisfaction
- Analyze CSAT trends across channels, product categories, agents, and cities
- Engineer meaningful features for machine learning
- Handle missing values and invalid data systematically
- Build and compare multiple classification models
- Address class imbalance effectively
- Evaluate different target formulations
- Select the most operationally useful prediction problem
- Recommend a responsible deployment strategy

## Dataset Information

The dataset contains approximately **85,907 customer support records** from Flipkart customer service interactions.

The dataset includes information such as:

- Support channels
- Issue categories and sub-categories
- Product categories
- Handling and response times
- Agent, supervisor, and manager details
- Customer cities
- Order-related attributes
- CSAT survey responses

Sensitive or proprietary information is not exposed in this repository documentation.

## Exploratory Data Analysis

The EDA phase focused on understanding customer satisfaction patterns and identifying operational areas linked with lower CSAT scores.

Key findings include:

- The majority of customers reported high satisfaction.
- Low CSAT responses still represented a meaningful business risk segment.
- Handling time and response delays influenced customer satisfaction.
- Some support channels performed better than others.
- Certain product categories showed comparatively lower satisfaction scores.
- Agent tenure, shift, and issue category showed visible differences in CSAT patterns.
- Several fields had high missingness, requiring careful interpretation and preprocessing.

The analysis included visualizations for:

- CSAT score distribution
- Channel-wise satisfaction
- Product category satisfaction
- Handling time vs CSAT
- Agent performance
- City-level trends
- Daily CSAT trends
- Feature correlations

## Feature Engineering

Several ML-ready features were engineered from the raw dataset to improve predictive modeling.

Engineered features include:

- Response time in minutes
- Response time buckets
- Issue hour
- Issue weekday
- Survey weekday
- Weekend issue indicator
- Same-day response flag
- Response delay flags
- Customer remark availability flag
- Order ID availability flag
- Agent interaction count
- Supervisor interaction count
- Manager interaction count
- City interaction count

Identifier and leakage-prone columns such as unique IDs, order IDs, raw timestamps, and target-derived fields were removed from the modeling feature set where appropriate.

## Machine Learning Models Evaluated

The following classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Additional models such as **XGBoost** and **LightGBM** were handled gracefully in the notebook if the required libraries were unavailable in the runtime environment.

## Model Evaluation Strategy

Accuracy alone was not sufficient for this project because the CSAT target was highly imbalanced. Most customers belonged to the high satisfaction group, meaning a model could achieve misleadingly high accuracy by mostly predicting the majority class.

To evaluate models more responsibly, the following metrics were used:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score
- Weighted F1 Score
- Classification Reports
- Confusion Matrices

**Macro F1 Score** was treated as an important metric because it gives equal importance to all classes, including minority dissatisfaction classes.

## Target Formulation Analysis

Three target formulations were evaluated:

1. **Low / Neutral / High**
   - Low: CSAT 1-2
   - Neutral: CSAT 3
   - High: CSAT 4-5

2. **High vs Non-High**
   - High: CSAT 4-5
   - Non-High: CSAT 1-3

3. **Low vs Not-Low**
   - Low: CSAT 1-2
   - Not-Low: CSAT 3-5

The 3-class formulation was useful for analysis, but the Neutral class was highly imbalanced and difficult for models to predict reliably.

The final recommended formulation is **Low vs Not-Low** because it is the most operationally useful. It directly supports the business objective of identifying dissatisfied customers who may require intervention.

## Final Recommendation

The final recommendation is to deploy a **Random Forest classifier** using the **Low vs Not-Low** target formulation.

This formulation is recommended because:

- It focuses directly on dissatisfied customers.
- It avoids instability caused by the very small Neutral class.
- It provides a clear business action: identify interactions that may require escalation or service recovery.
- It balances predictive performance with operational usefulness.

The model should be used as a **human-in-the-loop decision support system**, not as a fully automated decision-making tool.

## Production Considerations

Before real-world deployment, the following production considerations should be addressed:

### Monitoring

Monitor model performance continuously using:

- Low-class recall
- False negative rate
- Macro F1 Score
- Prediction distribution
- Channel-wise prediction trends
- Escalation and recovery outcomes

### Retraining

The model should be retrained periodically as support processes, customer behavior, product mix, and operational policies change.

A monthly retraining cycle is recommended initially, with more frequent retraining during major sale events or policy changes.

### Data Drift Detection

Data drift should be monitored across important features such as:

- Support channel mix
- Issue category distribution
- Response time patterns
- Product category mix
- Agent tenure distribution
- Missing value rates
- Predicted Low-CSAT probability distribution

### Fairness Considerations

The model should not be used to unfairly penalize agents, cities, shifts, or teams. Some groups may handle more complex cases or operate under different constraints.

Fairness checks should compare prediction and error rates across:

- Support channels
- Agent shifts
- Tenure buckets
- Supervisors
- Major city groups

### Limitations of the Current Approach

Current limitations include:

- The dataset covers a limited time period.
- Some important fields have high missingness.
- Customer remarks were not modeled using NLP.
- Advanced libraries such as XGBoost, LightGBM, and SHAP may require additional runtime setup.
- The model identifies associations, not causal relationships.
- A time-based validation strategy should be used before real deployment.

## Repository Structure

```text
.
├── Sample_EDA_Submission_Template.ipynb
├── Sample_ML_Submission_Template.ipynb
├── README.md
└── Customer_support_data.csv (if permitted)
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Open the Notebook

Open the notebook using either:

- Jupyter Notebook
- JupyterLab
- Google Colab
- VS Code with Jupyter extension

### 3. Upload or Place the Dataset

Ensure the dataset file is available in the project directory:

```text
Customer_support_data.csv
```

If using Google Colab, upload the dataset manually before running the notebook.

### 4. Run All Cells

Run the notebook cells sequentially from top to bottom.

The notebook includes:

- Data loading
- EDA
- Feature engineering
- ML preprocessing
- Model training
- Model evaluation
- Target formulation comparison
- Production deployment recommendations

## Results

Key project results include:

- Random Forest was the strongest overall model among the evaluated classifiers.
- The 3-class CSAT formulation was useful analytically but struggled due to the small Neutral class.
- The **Low vs Not-Low** formulation provided the best operational value.
- The model can help Flipkart proactively identify dissatisfied customers and prioritize support interventions.
- Human review is recommended before taking operational action based on predictions.

## Future Improvements

Potential future improvements include:

- Applying NLP techniques to customer remarks
- Building real-time prediction pipelines
- Testing advanced ensemble models
- Adding calibrated probability thresholds
- Using time-based validation
- Developing human-in-the-loop escalation workflows
- Integrating model monitoring dashboards
- Expanding fairness and bias audits
- Improving missing data capture at the source

## Author

**Author:** Aayush Kulkarni

Internship Capstone Project  
Labmentix Internship
