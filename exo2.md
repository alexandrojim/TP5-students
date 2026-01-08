# Explainable AI (XAI) with SHAP Values

## Objectives
* Implement a high-performance "Black Box" model (XGBoost).
* Use **SHAP (SHapley Additive exPlanations)** to break down model predictions.
* Differentiate between **Global Interpretability** (how the model works) and **Local Interpretability** (why a specific decision was made).

## Prerequisites
You will need a Python environment with the following libraries:
```bash
pip install numpy pandas matplotlib xgboost shap scikit-learn
```

## 1. The Scenario: The Automated Loan Officer
You are a Data Scientist at a fintech company. You have built a gradient-boosted tree model to predict whether an applicant earns >$50k/year (a proxy for loan eligibility). While the model is accurate, the legal department requires you to explain why a loan is denied to ensure the process is fair and transparent.

Data Preparation & Model Training: We will use the Census Income dataset.

```python
import xgboost
import shap
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the dataset
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a "Black Box" XGBoost model
model = xgboost.XGBClassifier(n_estimators=100, max_depth=4).fit(X_train, y_train)

# 3. Calculate Accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")
```

## 2. Global Interpretability: The "Big Picture"
Global interpretability helps us understand the model's overall logic. We want to know which features are generally the most important.

Task: Create a SHAP Summary Plot.

```python
# Initialize the SHAP Explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Visualize global feature importance
shap.summary_plot(shap_values, X_test)
```

Student Analysis:
- Which feature is the most influential?
- Look at the "Color" (Feature Value). Does a higher value of your top feature increase or decrease the probability of high income?

## 3. Local Interpretability: The Individual Case
A customer named "Applicant #10" was denied. You must provide a specific reason.

Task: Create a Force Plot for a single prediction.

```python
# Visualize the prediction for the 10th person in the test set
# The 'base value' is the average model output. 
# The arrows show how each feature pushes the prediction away from the average.
shap.plots.force(shap_values[10])
```

Student Analysis:
- What was the model's raw output score for this person?
- Which specific feature was the primary reason for their score being higher or lower than the average?

## 4. Detecting Bias and Interaction
SHAP can reveal hidden biases, such as how `Age` or `Relationship Status` affects the model in ways that might be discriminatory.

Task: Create a Dependence Plot to see how `Age` interacts with `Education`.

```python
# Interaction between Age and Education-Num
shap.plots.scatter(shap_values[:, "Age"], color=shap_values[:, "Education-Num"])
```

# Lab Report Requirements
1. Model Performance: Report your XGBoost accuracy.
2. Global Insights: Based on the Summary Plot, list the top 3 features the bank uses to determine "Creditworthiness."
3. The "Denial Letter": Imagine Applicant #10 was rejected. Based on the Force Plot, write a 2-sentence explanation to the customer explaining the primary factors behind the decision.
4. Critical Thinking: Identify a feature in the dataset that might be a "proxy" for a protected characteristic (like race or gender). How would you use SHAP to prove if the model is biased?
