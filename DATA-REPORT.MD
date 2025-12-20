# DATA REPORT

## 1. DATASET INFORMATION

### Dataset Source

[* **IMDB Non-Commercial Datasets**](https://developer.imdb.com/non-commercial-datasets/)


### Raw Dataset Files Used

* `title.basics.tsv`
* `title.ratings.tsv`

### Dataset Description

| File Name           | Description                | Columns |
| ------------------- | -------------------------- | ------- |
| `title.basics.tsv`  | Core title metadata        | 9       |
| `title.ratings.tsv` | User ratings & vote counts | 3       |

### Common Identifier

* `tconst` (unique title identifier)


## 2. DATASET PROCESSING FLOW (RAW → PROCESSED)

The preprocessing pipeline transforms raw IMDB TSV files into a single clean dataset:
`src/data/processed/final.csv`

### Step 1: Raw Data Loading

* Both TSV files are loaded using:

  * Tab (`\\t`) separation
  * `\\N` treated as missing values
* `tconst` is explicitly enforced as a string
* Rows with missing or invalid `tconst` values are rejected

**Reasoning:**
`tconst` is the primary key. Any corruption here invalidates joins and downstream analysis.


### Step 2: Movie Filtering

* Dataset filtered to retain only:

  ```text
  titleType == "movie"
  ```
* Non-movie titles (TV shows, shorts, episodes, etc.) are removed

**Reasoning:**
The modeling task focuses exclusively on movies to avoid heterogeneous rating behavior.


### Step 3: Dataset Merge

* `title.basics` and `title.ratings` are merged using:

  ```python
  how="left"
  validate="many_to_one"
  ```
* This ensures:

  * One rating row per movie
  * No accidental duplication

**Result:**

* Initial merged dataset contains **11 features**

### Step 4: Removing Redundant Columns

The following columns are removed:

* `titleType`
* `endYear`

**Justification:**

* `titleType` becomes redundant after filtering
* `endYear` contains only missing values for movies

### Step 5: Missing Target Value Records Handled

* Rows with missing `averageRating` are removed

**Reasoning:**

* Ratings are required to:

  * define our target variable for upcoming ML tasks

### Step 6: Temporal Feature Cleaning (`startYear`)

* Converted to numeric
* Rows with invalid or missing years are dropped
* Cast to integer

### Step 7: Runtime Outlier Handling

* `runtimeMinutes`:

  * Converted to numeric
  * Values ≤ 0 removed
  * Log transformation applied
  * Z-score filtering applied (`|z| ≤ 4`)

### Step 8: Vote Count Outlier Handling

* `numVotes`:

  * Converted to numeric
  * Values ≤ 0 removed
  * Log transformation applied
  * Z-score filtering applied (`|z| ≤ 4`)

### Step 9: Final Dataset Export

* Cleaned dataset is saved as:

  ```
  src/data/processed/final.csv
  ```


## 3. FINAL DATASET SUMMARY

### Retained Features

| Feature Name     | Description            |
| ---------------- | ---------------------- |
| `tconst`         | Movie identifier       |
| `primaryTitle`   | Movie title            |
| `originalTitle`  | Original release title |
| `isAdult`        | Adult content flag     |
| `startYear`      | Release year           |
| `runtimeMinutes` | Movie runtime          |
| `genres`         | Genre labels           |
| `averageRating`  | Mean user rating       |
| `numVotes`       | Number of votes        |

### Target Variable (Future Tasks)
* `averageRating`

## 4. DATA QUALITY CONTROLS APPLIED

* Primary key validation (`tconst`)
* Strict movie-only filtering
* Missing target elimination
* Log-based outlier detection + Z score Transform
* Numeric type conversion for numeric data columns

## 5. OUTPUT ARTIFACT

```
src/data/processed/final.csv
```

This processed dataset will be used for the following tasks:

* Exploratory Data Analysis
* Feature Engineering
* Category encoding