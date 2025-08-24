# Term Deposit Subscription Prediction

##  Project Overview
This project focuses on predicting whether a client will subscribe to a term deposit based on marketing campaign data.  
The dataset contains information about client demographics, previous campaign contacts, and other relevant attributes.

---

##  Project Structure
- **`Term Deposit Subscription Prediction.ipynb`** → Main Jupyter notebook containing the complete data analysis and modeling process  
- **`app.py`** → Streamlit web application for interactive predictions  
- **`voting_model_personal_loan.pkl`** → Serialized trained model for deployment  
- **`bank-full.csv`** → The dataset used for training and evaluation  

---

##  Key Features

###  Data Analysis
- Comprehensive exploratory data analysis (EDA)  
- Visualization of feature distributions and relationships  
- Handling of class imbalance in the target variable  

###  Modeling Approach
Implemented multiple machine learning algorithms:  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Trees  
- Support Vector Machines (SVM)  
- Random Forest  
- Extra Trees  
- AdaBoost  
- Gradient Boosting  
- XGBoost  

###  Handling Class Imbalance
Techniques applied to address the significant class imbalance non of these help in improving the accuracy:  
- Random UnderSampling  
- SMOTE (Synthetic Minority Over-sampling Technique)  
- Random OverSampling  

###  Model Evaluation
Comprehensive evaluation metrics including:  
- Classification reports  
- Confusion matrices  
- Accuracy scores  
- Hyperparameter tuning using GridSearchCV  

---

##  Deployment
- Streamlit web application for making predictions  
- Serialized model (`model.pkl`) for easy deployment

## Live app
https://term-deposit-subscription-prediction-hkxmgu7zmlxsfveatk2bx9.streamlit.app/
