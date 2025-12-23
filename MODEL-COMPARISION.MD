# MODEL COMPARISON

This document presents a comparative evaluation of multiple machine learning models trained to predict **ordinal movie rating classes** derived from average ratings.

## 1. Models Trained

The following models were trained and evaluated using the same preprocessing pipeline (`StandardScaler`) and identical feature sets. The configurations below reflect the **exact hyperparameters used during training**.

### 1. Logistic Regression (One-vs-Rest)

- **Estimator:** `LogisticRegression`
- **Multiclass Strategy:** `OneVsRestClassifier`
- **Configuration:**
  - `max_iter`: 1000
  - `solver`: lbfgs

### 2. Random Forest

- **Estimator:** `RandomForestClassifier`
- **Configuration:**
  - `n_estimators`: 200
  - `max_depth`: 10
  - `min_samples_leaf`: 5
  - `random_state`: 42
  - `n_jobs`: -1

### 3. XGBoost

- **Estimator:** `XGBClassifier`
- **Configuration:**
  - `n_estimators`: 200
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `eval_metric`: mlogloss
  - `use_label_encoder`: False
  - `random_state`: 42

### 4. Neural Network (Multi-Layer Perceptron)

- **Estimator:** `MLPClassifier`
- **Configuration:**
  - `hidden_layer_sizes`: (64, 32)
  - `alpha`: 0.0001
  - `max_iter`: 300
  - `random_state`: 42

## 2. Evaluation Metrics

The following metrics were used to evaluate model performance:

- **Accuracy**
- **Precision (Weighted)**
- **Recall (Weighted)**
- **F1 Score (Weighted)**
- **ROC-AUC (Multiclass, One-vs-Rest, Weighted)**
- *Model selection criterion: The model with the highest **cross-validated ROC-AUC score** was selected as the best model.*  

## 3. Cross-Validation Strategy

- **Technique:** Stratified K-Fold Cross Validation  
- **Number of folds:** 5  
- **Stratification:** Preserves class distribution across folds  

For each fold:
1. Model is trained on 4 folds
2. Validated on 1 fold
3. Metrics are computed
4. Final CV metrics are averaged across folds

## 4. Cross-Validation Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|---------|-----------|--------|----|--------|
| Logistic Regression (OvR) | 0.404 | 0.382 | 0.404 | 0.382 | 0.665 |
| Random Forest | 0.415 | 0.400 | 0.415 | 0.388 | 0.692 |
| **XGBoost** | **0.423** | **0.426** | **0.423** | **0.403** | **0.702** |
| Neural Network (MLP) | 0.419 | 0.420 | 0.419 | 0.398 | 0.694 |

## 5. Best Model Selection

- **Selected Model:** XGBoost  
- **Selection Basis:** Highest cross-validated ROC-AUC score  
- **Test Set ROC-AUC:** 0.702  

XGBoost demonstrated the strongest ability to separate ordinal rating classes and consistently outperformed other models across most metrics.

## 6. Confusion Matrix (Best Model)

A confusion matrix was generated for the **XGBoost model** using the held-out test set.

- Visualizes class-wise prediction performance
- Highlights confusion between adjacent ordinal classes
- Saved at: `results/confusion_matrix.png`
