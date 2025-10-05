
# Machine Learning Project

This repository contains a **machine learning project** completed as part of a university course.  
This project builds a **complete ML pipeline** for predicting the sentiment of social media posts, including data collection, preprocessing, feature engineering, model training, evaluation, and clustering.

---

## 📌 Project Overview

This project implements a **full ML pipeline** for predicting sentiment (positive/negative) of social media posts, covering data collection, preprocessing, modeling, evaluation, and final prediction.

- **Goal:** Predict post sentiment based on textual and contextual features.  
- **Dataset:** Static dataset of posts (text, user info, likes, shares).  
- **Pipeline:**
  - **Data Collection & Sensing:** Collected static data from posts.  
  - **Exploratory Data Analysis (EDA):** Explored distributions and correlations to uncover sentiment-related patterns.  
  - **Preprocessing:** Cleaned missing values, encoded categorical variables, and generated new features.  
  - **Feature Extraction:** Extracted semantic and syntactic features using TF-IDF and N-grams.  
  - **Feature Selection:** Selected top 25 features using Fisher Score to reduce dimensionality and overfitting.  
  - **Validation:** Applied 10-fold cross-validation with strict separation to avoid data leakage.  
  - **Model Training:** Trained Decision Tree, SVM, and MLP models; tuned hyperparameters using RandomizedSearchCV.  
  - **Evaluation:** Analyzed confusion matrices, ROC curves, and feature importance for each model.  
  - **Improvements:** Suggested enhancements in feature design and user profiling to boost performance.  
  - **Final Prediction:** Predicted sentiment on a hold-out test set using the complete ML pipeline.

---

## 🧪 Final Results

| Model            | AUC (Validation) | Notes |
|------------------|-----------------|-------|
| Decision Tree    | 0.958           | ✅ Selected final model |
| MLP              | 0.965           | Best raw performance |
| SVM              | 0.900           | Lower accuracy |

- **Final chosen model:** Decision Tree — best trade-off between performance and interpretability.

---

## 🧰 Libraries & Tools Used

| Category           | Libraries / Tools                              |
| ------------------ | ---------------------------------------------- |
| Data Handling      | `pandas`, `numpy`                              |
| Visualization      | `matplotlib`, `seaborn`                        |
| Modeling & ML      | `scikit-learn`                                 |
| Feature Extraction | TF-IDF, N-grams (via `sklearn`)                |
| Clustering         | K-Medoids (via `scikit-learn-extra` or custom) |
| Environment        | Python 3.8+, Jupyter Notebook                  |
