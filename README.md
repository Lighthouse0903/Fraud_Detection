
# Credit Card Fraud Detection System

## Overview
This project focuses on developing a supervised machine learning system to detect fraudulent financial transactions. By analyzing and processing real-world financial transaction data, the system aims to differentiate between legitimate and fraudulent activities effectively.

## Objectives
### General Objective
- Build and evaluate the effectiveness of various data science methodologies in detecting fraudulent transactions, leading to an optimized model for real-world applications.

### Specific Objectives
- Understand and analyze fraud patterns in financial data.
- Collect and preprocess relevant datasets, handling imbalanced data issues.
- Apply machine learning algorithms such as Logistic Regression, Random Forest, and Gradient Boosting to detect fraud.
- Evaluate models using metrics like Accuracy, Precision, Recall, F1-score, and ROC AUC.
- Propose a practical solution with high detection accuracy and minimal false positives/negatives.

## Dataset
- **Source**: [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
- **Description**: The dataset contains 11 columns and over 6 million rows, including transaction details such as:
  - `step`: Transaction time
  - `type`: Transaction type (e.g., transfer, cash-out)
  - `amount`: Transaction amount
  - `oldbalanceOrg`, `newbalanceOrig`: Sender's balance before and after the transaction
  - `oldbalanceDest`, `newbalanceDest`: Receiver's balance before and after the transaction
  - `isFraud`: Binary indicator (1 for fraud, 0 for legitimate)

## Methodology
1. **Data Preprocessing**:
   - Cleaned and normalized features like `amount`, `oldbalanceOrg`, `newbalanceOrig` to handle skewness and outliers using log transformations.
   - Balanced the dataset by down-sampling the majority class (non-fraudulent transactions) to address imbalanced data.
   - Split the balanced dataset into training (60%), validation (20%), and testing (20%) sets.

2. **Exploratory Data Analysis (EDA)**:
   - Identified outliers and data distribution using Pearson and Spearman correlation matrices.
   - Visualized relationships between features and their contributions to fraud detection.

3. **Modeling**:
   - Trained and compared three machine learning models: Logistic Regression, Random Forest, and Gradient Boosting.
   - Evaluated models using metrics like Accuracy, Precision, Recall, F1-score, and ROC AUC.

4. **Hyperparameter Optimization**:
   - Used Hyperopt with TPE (Tree-structured Parzen Estimator) for optimizing the Random Forest model.

5. **Web Interface**:
   - Deployed the final Random Forest model in a web application for real-time fraud detection.

## Results
- The **Random Forest Classifier** outperformed other models with the following metrics:
  - **Accuracy**: 99.11%
  - **Precision**: 98.48%
  - **Recall**: 98.84%
  - **F1-score**: 98.66%
  - **ROC AUC**: 99.04%

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Lighthouse0903/Fraud_Detection.git
   cd Fraud_Detection
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Start the web application:
   ```bash
   python app.py
   ```

## Future Enhancements
- **Deep Learning**: Incorporate advanced models like CNNs and RNNs for complex fraud patterns.
- **Real-Time Processing**: Utilize tools like Apache Kafka and Spark Streaming for real-time fraud detection.
- **Big Data Integration**: Implement platforms like Hadoop or Elasticsearch for large-scale transaction analysis.

## References
1. Ian H. Witten, Eibe Frank, Mark A. Hall: “Data Mining: Practical Machine Learning Tools and Techniques”.
2. Christopher M. Bishop: "Pattern Recognition and Machine Learning".
3. Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org/stable/)

---
