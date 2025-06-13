# Customer Churn Prediction Using SQL + Random Forest

This project analyzes customer churn behavior using a combination of SQL, pandas, and machine learning (Random Forest Classifier). We use a marketing dataset (`marketing_campaign.csv`) enriched with synthetic behavioral attributes such as contract type, monthly spend, and last active date to simulate churn behavior. The model predicts whether a customer is likely to churn based on their profile and interaction history.

The project uses the publicly available dataset: **`marketing_campaign.csv`**, originally containing demographic and marketing engagement data of customers.

### Original Features
- `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer`, `Recency`, and various product purchase metrics.

### Added Synthetic Features
To enable churn prediction, the following additional fields were generated:
- `customer_id`: Duplicate of `ID`
- `contract_type`: Randomly assigned as 'Month-to-Month', 'One Year', or 'Two Year'
- `monthly_spend`: Normally distributed around 60 units
- `last_active`: Randomly generated datetime between Jan 2021 and Jan 2024


- **Python 3.x**
- **SQLite3** (for relational querying)
- **pandas** (for data manipulation)
- **scikit-learn** (for modeling)
- **ipython-sql** (to run SQL queries inside Jupyter Notebook)

Results

              precision    recall  f1-score   support

           1       1.00      1.00      1.00       448

    accuracy                           1.00       448
   macro avg       1.00      1.00      1.00       448
weighted avg       1.00      1.00      1.00       448
The model returned 100% accuracy, but this result is misleading.

Upon inspection, all records were labeled as churned = 1 based on the synthetic last_active values. 
This leads to a lack of label variability, causing the model to trivially predict "1" for all test cases.

Even when a model achieves perfect accuracy, it is essential to inspect the data balance.
This project demonstrates how to build a complete SQL + ML pipeline, but also highlights the importance of realistic and diverse labels in supervised learning.
In a real-world scenario, further refinement is needed such as:
-Sourcing or generating a mix of churned and active customers.
-Applying techniques for imbalanced classification if class imbalance persists.
