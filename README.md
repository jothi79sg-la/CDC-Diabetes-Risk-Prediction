# CDC-Diabetes-Risk-Prediction
### Predicting Diabetes Risk to Support Early Healthcare Intervention

**Project Type:** Machine Learning / Healthcare Analytics / Explainable AI

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, XGBoost, SHAP

**Dataset:** CDC Diabetes Health Indicators

**Champion Model:** XGBoost

**1. Executive Summary**

Diabetes is a major chronic health condition associated with long-term complications and increasing healthcare costs. Healthcare organisations may have limited resources for comprehensive diabetes screening, creating a need to prioritise individuals who may benefit from further screening or preventive intervention.

This project uses the CDC Diabetes Health Indicators dataset to develop a machine learning classification solution for identifying individuals belonging to the positive diabetes-risk target class.
The project follows an end-to-end machine learning workflow, including data quality assessment, exploratory data analysis, data cleaning, class-imbalance handling, model development, model evaluation and Explainable AI.

**Four classification models were compared:**

Logistic Regression

Decision Tree

Random Forest

XGBoost

XGBoost was selected as the **champion model** based on its stronger performance for identifying positive target cases. The project prioritises Recall, F1-score, ROC-AUC and MCC because missing potential positive cases is an important consideration for a screening-prioritisation use case.

SHAP was also applied to explain the model's predictions and identify the features contributing to individual risk predictions.

This project is intended as a machine-learning decision-support and screening-prioritisation concept. It is not a clinical diagnostic system, and final healthcare decisions should remain with qualified professionals.

**2. Business Problem:**

Diabetes is one of the leading chronic diseases worldwide and is associated with significant healthcare costs, long-term complications, and reduced quality of life. Healthcare organisations often have limited resources for comprehensive diabetes screening. By using machine learning to analyse demographic, lifestyle, and health-related indicators, healthcare programmes can use predicted risk to prioritise individuals for further screening or preventive intervention.

**3. Project Objectives:**

a. Develop a machine learning model to predict diabetes risk using demographic, lifestyle, and health-related indicators.

b. Compare the performance of multiple machine learning algorithms and identify the most suitable prediction model.

c. Apply Explainable Artificial Intelligence to interpret model predictions and identify the most influential risk factors.

d. Recommend a practical deployment approach that supports data-driven decision-making for preventive healthcare programs.

**4. Stakeholder / Persona:**

**Persona:** Mr.Lim

**Role:** Population Healthcare Programme Manager

**Goals:**

Improve early identification of individuals at high risk of diabetes.

Increase the effectiveness of community screening programmes.

Allocate preventive healthcare resources efficiently.

Reduce long-term diabetes complications.

**Pain Points:**

Limited healthcare resources.

Increasing prevalence of diabetes.

Difficulty identifying high-risk individuals before diagnosis.

Rising healthcare costs associated with chronic disease.

**5. JOBS-TO-BE-DONE(JTBD):**

When planning preventive healthcare programmes, I want to identify individuals who are at high risk of diabetes so that healthcare resources can be allocated efficiently, preventive interventions can be delivered earlier, and long-term diabetes complications can be reduced.

**6. DataSet Overview:**

**Dataset Summary:**

Total Records: 253,680

Duplicate Records: 24,206 (9.54%)

Features: 22 (21 Predictors +1 Target)

Target Variable: Diabetes Binary

Missing values: None

Original Data Type: All variables stored as int64.

**Variable Classification:**

Although all variables were originally stored as integers (int64), they were classified according to their meaning and measurement scale rather than their storage data type.

The variables were grouped into:

    Binary: 15 variables

    Ordinal: 4 variables

    Numerical: 3 variables

This classification was used to guide data quality checks, preprocessing, and modelling decisions.

**Target Variable:**

0 = No Diabetes

1 = Prediabetes or Diabetes

**Target distribution:**

<img width="395" height="274" alt="image" src="https://github.com/user-attachments/assets/4a54f232-4e2a-4875-a584-d60c5b0fd0e1" />

The dataset contains two target classes: **No Diabetes and Prediabetes/Diabetes**. The target distribution is imbalanced, with the positive class representing a smaller proportion of observations. This imbalance is important because a model could achieve high accuracy while still missing many positive cases. Therefore, class imbalance handling and Recall-based evaluation are considered later in the modelling stage.

**7. Data Quality:**

**Dataset Size Before and After Duplicate Removal:**

<img width="455" height="275" alt="image" src="https://github.com/user-attachments/assets/142866ae-5c46-49a9-9210-1d220b70f002" />

**Missing values:**

No missing values were detected across the dataset, so no imputation was required.

**Outliers:**

Potential outliers were identified in BMI, MentHlth, and PhysHlth using the IQR method.

<img width="674" height="217" alt="image" src="https://github.com/user-attachments/assets/b8314458-e3f2-40f5-bdae-3124df250600" />

Potential outliers were identified for investigation rather than automatically removed, as extreme values may represent 
genuine patient observations.

**8. Exploratory Data Analysis:**

**Diabetes Status by General Health:**

<img width="451" height="241" alt="image" src="https://github.com/user-attachments/assets/c0fd24f7-0301-40ee-bbe2-c32e135ead14" />

The distribution of diabetes status varies across General Health categories. Individuals reporting poorer general health show a different distribution of the positive Prediabetes/Diabetes class compared with individuals reporting better general health. This suggests that General Health contains useful information for distinguishing between the target classes.

**BMI Distribution by Diabetes Status:**

<img width="398" height="244" alt="image" src="https://github.com/user-attachments/assets/6d50d6e4-0009-4020-883b-9ba7af8ad64e" />

The BMI distributions differ between the two diabetes target groups. The Prediabetes/Diabetes group shows a higher BMI distribution compared with the No Diabetes group.BMI was also the dominant feature in the XGBoost built-in feature-importance analysis, indicating that the model relied heavily on BMI when generating predictions.
However, feature importance reflects model behaviour and does not establish that BMI alone causes diabetes.

**Diabetes Status by Age Group:**

<img width="455" height="244" alt="image" src="https://github.com/user-attachments/assets/6f9f6726-ac70-4b58-a6e8-d3c8eb43d6ce" />

The proportion of individuals in the Prediabetes/Diabetes class varies across age categories. The positive-class proportion generally becomes more prominent across the older age categories. Age was also identified as an important predictor in the subsequent model interpretation analyses.

**Diabetes Status by High Blood Pressure:**

<img width="398" height="245" alt="image" src="https://github.com/user-attachments/assets/4a78e91d-09b9-4a36-aa39-6c4248055648" />

The distribution of the diabetes target differs between individuals with and without high blood pressure. The proportion of the Prediabetes/Diabetes class is higher among individuals reporting HighBP.HighBP was also identified as an important predictor in the model interpretation stage.

**Physical Health by Diabetes Status:**

<img width="412" height="248" alt="image" src="https://github.com/user-attachments/assets/2ca7d74d-9c37-411d-8d31-8163791dafee" />

The distribution of reported physical-health days differs between the two diabetes target groups. The Prediabetes/Diabetes group shows greater variation in reported physical-health days.PhysHlth was also one of the most influential variables in the subsequent XGBoost feature-importance and SHAP analyses.

Overall, the EDA suggests that diabetes classification is associated with a combination of demographic, physical-health and health-status indicators rather than a single variable. These observations informed the subsequent machine learning modelling and feature-interpretation stages.

**9. Data Preparation Summary:**

<img width="389" height="125" alt="image" src="https://github.com/user-attachments/assets/6976c5fe-0845-45d7-be8f-65bbc458bc18" />

The dataset was cleaned and structured for modelling while preserving meaningful health-related information and avoiding unnecessary transformations.

**10. Class Imbalance:**

The target variable contains two classes:

0 — No Diabetes
1 — Prediabetes or Diabetes

The target distribution is imbalanced, with the positive class representing a smaller proportion of observations.
This creates an important modelling challenge because a model could achieve relatively high Accuracy by favouring the majority class while still missing many positive cases. Therefore, class-imbalance strategies were evaluated before selecting the final modelling approach.

**Why Class Imbalance Matters:**

A false negative occurs when an individual belonging to the positive target class is predicted as negative.
From the business perspective, this could mean that a potentially high-risk individual is not prioritised for further screening. Therefore, the project places greater emphasis on:

**Recall**

**F1-score**

**ROC-AUC**

**MCC**

while treating Accuracy as a supporting metric rather than the only evaluation measure.

**Sampling Strategies Compared:**

Three approaches were evaluated:

Original Data

Random Oversampling

Synthetic Minority Oversampling Technique (SMOTE)

**Selected Strategy:**

Random Oversampling was selected as the preferred class-imbalance strategy. The approach increased representation of the minority Prediabetes/Diabetes class without creating synthetic feature values. It provided stronger minority-class detection while maintaining competitive overall model performance. The selected Random Oversampling strategy was subsequently used for the model-comparison stage.

**11. Machine Learning:**

Four machine learning classification algorithms were evaluated to determine which approach best supports the objective of identifying potential Prediabetes/Diabetes cases.

The four models were compared:

   **Logistic Regression**

   **Decision Tree**

   **Random Forest**

   **XGBoost**

   <img width="611" height="293" alt="image" src="https://github.com/user-attachments/assets/d28692e2-45c0-4c8c-a924-8843cde03bb7" />
   
 **Evaluation Metrics:**
 
The four models were evaluated using the same selected Random Oversampling strategy and the same evaluation metrics.

<img width="427" height="140" alt="image" src="https://github.com/user-attachments/assets/6401a6b1-d68d-4a25-b83a-7e913507067c" />

**Machine Learning Workflow:**

<img width="276" height="331" alt="image" src="https://github.com/user-attachments/assets/17af71bb-0cae-4b60-bcbe-273e040351d2" />

**12. Feature Importance:** 

Feature-importance analysis identifies the variables that contribute most to the XGBoost model's predictions.

**Top 10 XGBoost Importance:**

<img width="508" height="291" alt="image" src="https://github.com/user-attachments/assets/61611ade-8527-4538-92ac-87a8c35e0926" />

BMI is the dominant feature, with an importance score of 0.5299.

PhysHlth is the second most important feature (0.1091).

Smoker, Age and HighBP are also among the leading predictors.

The remaining top features contribute smaller but measurable importance.

The model therefore relies on a combination of physical health, demographic, and behavioural factors.

**Top 10 Permutation Feature Importance:**

<img width="400" height="296" alt="image" src="https://github.com/user-attachments/assets/e0faff20-6152-45c2-9080-50d451a986bf" />

GenHlth had the highest permutation importance (0.0374). 

BMI (0.0295) and Age (0.0235) were the next most influential features. 

HighBP and HighChol also contributed to model performance. 

Shuffling these features caused a greater reduction in predictive performance. 

The ranking provides a complementary view of feature importance beyond the XGBoost built-in importance.

**13. SHAP Explainability:**

** Global SHAP Feature Importance:**

<img width="378" height="272" alt="image" src="https://github.com/user-attachments/assets/7a7cac7a-c30f-43d8-8bb1-2dbfc36e6302" />

Stroke: strongest average contribution, so its value can substantially alter predicted probability.

PhysHlth: second strongest; variation in physical-health status contributes strongly to prediction.

BMI and Age: important signals that materially influence model output.

**SHAP Beeswarm Plot:**

<img width="371" height="235" alt="image" src="https://github.com/user-attachments/assets/5863e69d-5935-4ddd-aa6e-092eae8557cf" />

Stroke, PhysHlth, BMI and Age have the largest overall effects.

Higher PhysHlth values are predominantly on the positive side, while lower values appear more on the negative side.

Higher BMI and Age values generally appear more toward the positive SHAP side.

HighBP and Smoker also show noticeable positive contributions for higher feature values in this model.

**14. Business Insights:**

The analysis indicates that **BMI, physical health, general health,blood pressure, age and lifestyle factors** are important signals in predicting the Prediabetes/Diabetes target class.

  **Healthy Weight & Physical Activity:** Support preventive programmes focused on healthy weight and physical activity.
 
  **Overall Health:** Consider general and physical health when prioritising individuals for further screening.
   
  **Cardiometabolic Health:** Include blood-pressure monitoring in preventive health initiatives.
  
  **Lifestyle:** Strengthen lifestyle-support initiatives such as smoking-cessation programmes.
  
  **Holistic Screening:** Use multiple health and demographic indicators rather than relying on a single risk factor.

  These are model-based associations, **not clinical diagnoses or evidence**.

**15. Recommendations:**
       
  **Targeted Screening:** Use the model to help prioritise individuals who may benefit from further diabetes screening and            professional assessment.

  **Healthy Lifestyle:** Strengthen preventive programmes focused on healthy weight, physical activity and nutrition.

  **Cardiometabolic Health:** Encourage appropriate blood-pressure monitoring and broader preventive health checks.

  **Lifestyle Risk Reduction:** Support smoking cessation and other healthy lifestyle initiatives where appropriate.

  **Holistic Assessment:** Consider multiple factors such as BMI, physical health, general health, age, blood pressure and lifestyle rather than relying on a single indicator.

  **Explainable Decision Support:** Use XGBoost and SHAP to support transparent screening prioritisation, while keeping healthcare professionals involved in final decisions.

  **Clinical Validation:** Further validate the model on appropriate new populations before considering real-world healthcare deployment.

  **Overall Recommendation:** Combine data-driven screening prioritisation with preventive healthcare and professional clinical assessment rather than using the model as a standalone diagnostic tool.

