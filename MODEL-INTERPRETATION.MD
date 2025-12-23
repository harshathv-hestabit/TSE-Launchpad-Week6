# MODEL INTERPRETATION

## Model Under Interpretation

* **Algorithm**: XGBoost (Multiclass Classification)
* **Objective**: `multi:softprob`
* **Evaluation Metric**: Weighted F1-score (during tuning)
* **Classes**: Ordinal rating buckets (0 → 4)

### Tuned Hyperparameters (Optuna)

```json
{
  "n_estimators": 233,
  "max_depth": 5,
  "learning_rate": 0.1669,
  "subsample": 0.7889,
  "colsample_bytree": 0.8908,
  "min_child_weight": 2,
  "gamma": 0.5137,
  "reg_alpha": 0.5296,
  "reg_lambda": 2.8915
}
```

The final model is wrapped inside a pipeline: `StandardScaler → XGBoostClassifier`

## SHAP

SHAP provides:

* **Consistent feature attributions**
* **Local explanations** (per prediction)
* **Global explanations** (overall feature importance)
* Strong theoretical grounding in **game theory**

For tree-based models, `TreeExplainer` computes SHAP values efficiently and exactly.


## SHAP Summary Plots

Each SHAP summary plot corresponds to **one target class**.

### Axes Explanation

* **Y-axis**: Features (sorted by importance)
* **X-axis**: SHAP value (impact on model output)

### Interpretation Rules

* **Positive SHAP value** → pushes prediction **towards** that class
* **Negative SHAP value** → pushes prediction **away from** that class
* Wider spread → higher impact on predictions

## Key Global Observations (Across All Classes)

### 1. `numVotes`

* Most influential feature across all classes
* High number of votes generally **increases confidence** in higher rating classes
* Low vote counts strongly push predictions toward lower ordinal classes

### 2. `runtimeMinutes`

* Longer runtimes tend to correlate with **higher-rated movies**
* Very short runtimes often push predictions downward

### 3. Genre-Based Features

Binary genre flags significantly affect predictions:

| Genre           | Typical Impact                            |
| --------------- | ----------------------------------------- |
| Drama           | Strong positive signal for higher classes |
| Documentary     | Polarizing; strongly class-dependent      |
| Horror          | Often negative for higher classes         |
| Animation       | Mixed but can boost mid-to-high classes   |
| Adult / Unknown | Strong negative contribution              |

## Class-wise Interpretation

### Class 0 – *Very Poor*

* High influence from:

  * Low `numVotes`
  * Presence of `Adult`, `Western`, or `Unknown`
* Absence of strong narrative genres (Drama, Biography)

### Class 1 – *Poor*

* Slightly higher `numVotes` than Class 0
* Genre signals still weak
* Horror and low-runtime titles frequently appear

### Class 2 – *Average*

* Balanced contribution from:

  * Moderate `numVotes`
  * Runtime around dataset mean
* Genres like Comedy, Romance, Crime contribute mildly

### Class 3 – *Good*

* Strong positive impact from:

  * High `numVotes`
  * Drama, Biography, Animation
* Runtime generally above average

### Class 4 – *Excellent*

* Dominated by:

  * Very high `numVotes`
  * Drama / Documentary dominance
* Low noise from negative genres