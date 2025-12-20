# FEATURE-ENGINEERING-DOC.md

## DAY 2 — FEATURE ENGINEERING & FEATURE SELECTION

## 1. OBJECTIVE

The objective of Day 2 is to transform the cleaned dataset produced on Day 1 into:

1. A **model-ready feature matrix**
2. A **ordinal classification target**
3. A **reduced feature subset** suitable for downstream classification models

## 2. INPUT DATA

### Source Dataset

```
src/data/processed/final.csv
```

This dataset has been processed to handle missing values and outliers.

## 3. TARGET VARIABLE ENGINEERING

### 3.1 Raw Target Source

* Column: `averageRating`
* Type: continuous (IMDB rating scale)

### 3.2 Derived Target

* Column: `rating_class`
* Type: **ordinal categorical**

### 3.3 Ordinal Encoding Strategy

The raw rating is discretized into five ordered classes using fixed thresholds:

| Rating Range | Encoded Label | Semantic Meaning |
| ------------ | ------------- | ---------------- |
| ≤ 4.5        | 0             | Very Poor        |
| 4.5 – 6.0    | 1             | Poor             |
| 6.0 – 7.0    | 2             | Average          |
| 7.0 – 8.0    | 3             | Good             |
| > 8.0        | 4             | Excellent        |

Implementation code:

```python
pd.cut(ratings, bins=[-inf, 4.5, 6.0, 7.0, 8.0, inf])
```

## 4. FEATURE ENGINEERING

### 4.1 Excluded Features

The following columns are intentionally excluded from modeling:

* `averageRating` (used only for label generation)
* `startYear` (removed to avoid temporal bias)
* Identifiers (`tconst`)

### 4.2 Numeric Features

Processing steps:

* Features scaled using `StandardScaler`

Final numeric feature set:

```
["runtimeMinutes", "numVotes"]
```

### 4.3 Categorical Features

#### Genres

* Multi-label categorical feature
* Original format: comma-separated string

Processing steps:

1. Missing values replaced with `"Unknown"`
2. Genres split on comma
3. Multi-hot encoded using dummy variables

Total genre-derived features: **28**


## 5. FEATURE MATRIX CONSTRUCTION

The final feature matrix is constructed as:

```
X = [numeric_features + genre_features]
y = rating_class
```

Final feature count:

* Numeric: 2
* Categorical: 28
* **Total: 30 features**

## 6. TRAIN–TEST SPLIT STRATEGY

### Split Configuration

* Train size: 70%
* Test size: 30%

### Stratification

```python
train_test_split(..., stratify=y)
```

## 7. OUTPUT ARTIFACTS — FEATURE ENGINEERING

```
src/data/processed/
├── X_train.csv
├── X_test.csv
├── y_train.csv
└── y_test.csv

src/features/
└── feature_list.json
```

### `feature_list.json`

Contains:

* Numeric feature names
* Categorical feature names
* Full feature schema
* Target column name
* Ordinal class definitions


## 8. FEATURE SELECTION

Feature selection is performed using **training data only**.

### 8.1 Correlation Filtering

* Threshold: `0.95`
* Removes redundant features

### 8.2 Mutual Information (Classification)

* Method: `mutual_info_classif`
* Top `30` features retained
* Measures dependency between features and ordinal labels

### 8.3 Recursive Feature Elimination (RFE)

* Estimator: `RandomForestClassifier`
* Class weighting: `balanced`
* Final features selected: `25`