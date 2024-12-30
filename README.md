# **Credit Fraud Detection Project**

## **Overview**
This project focuses on building a machine learning model to detect fraudulent financial transactions. By leveraging a large dataset with imbalanced classes, the project aims to identify fraudulent activity with high accuracy while minimizing false positives to ensure reliability in real-world applications.

---

## **Features**
- **Objective:** Detect and classify transactions as fraudulent or non-fraudulent.
- **Dataset Size:** Over 6 million transactions.
- **Techniques Used:**
  - Data preprocessing and feature engineering.
  - Machine learning algorithms: Random Forest, Logistic Regression, Gradient Boosting.
  - Hyperparameter optimization using Grid Search.
- **Performance:** Achieved an AUC score of 0.9904 with high precision, recall, and F1-score.

---

## **Technologies**
- **Programming Language:** Python
- **Libraries:**
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
- **Tools:**
  - Jupyter Notebook for model development and experimentation
  - Git for version control
  - Deployment as a Python-based interactive interface

---

## **Steps Involved**

### **1. Data Preparation**
- Cleaned raw data to handle missing values and outliers.
- Balanced class distribution using down-sampling.
- Transformed data with log normalization and clipping.

### **2. Exploratory Data Analysis (EDA)**
- Conducted analysis to identify trends and patterns in transactional data.
- Highlighted correlations and significant features impacting fraudulent activity.

### **3. Model Development**
- Compared multiple models for best performance:
  - Random Forest
  - Logistic Regression
  - Gradient Boosting
- Optimized hyperparameters for the Random Forest model using cross-validation.

### **4. Evaluation**
- Evaluated performance using metrics such as:
  - Precision: 98.48%
  - Recall: 98.84%
  - F1-Score: 98.66%

### **5. Deployment**
- Packaged the model as a `.pkl` file for easy integration.
- Created a user-friendly Python application for real-time fraud detection.

---

## **Key Achievements**
- Built a robust fraud detection model with state-of-the-art performance.
- Improved identification accuracy for fraudulent transactions, significantly reducing false positives.

---

