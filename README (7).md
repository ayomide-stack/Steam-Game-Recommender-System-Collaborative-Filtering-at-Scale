# Steam Game Recommender System — Collaborative Filtering at Scale

**ALS-based collaborative filtering recommender system built on Databricks using PySpark, MLlib, and MLflow.**

## Overview

This project builds a scalable recommender system using the Steam-200K dataset, which contains 200,000 user–game interaction records. An Alternating Least Squares (ALS) model is trained using PySpark's MLlib on Databricks Community Edition, with MLflow used to track experiments, tune hyperparameters, and log evaluation metrics.

## Tools & Stack

| Category | Stack |
|---|---|
| Language | Python (PySpark) |
| Platform | Databricks Community Edition |
| ML Framework | MLlib (ALS — Alternating Least Squares) |
| Experiment Tracking | MLflow (experiment logging, hyperparameter tuning, metric tracking) |
| Data Processing | PySpark DataFrames, SQL |
| Dataset | Steam-200K (user–game play behaviour) |

## Project Structure

```
steam-recommender-databricks/
├── TASK_2_RECOMMENDER_SYSTEM.html    # Exported Databricks notebook
├── README.md
```

> Note: This project was developed on Databricks Community Edition. The notebook is exported as HTML for portfolio viewing. To run interactively, import the notebook into a Databricks workspace and attach a Spark cluster.

## Dataset

The Steam-200K dataset contains:
- ~200,000 user–game interaction records
- Features: `user_id`, `game_name`, `behaviour` (play/purchase), `hours`
- Preprocessed to extract implicit feedback (play hours as interaction signal)

## Methodology

### 1. Data Ingestion & Preprocessing
- Loaded dataset into PySpark DataFrame
- Filtered to `play` interactions only (removing purchase-only records)
- Indexed string user IDs and game names to integer indices (required for ALS)
- Split into training and test sets

### 2. ALS Model
ALS (Alternating Least Squares) is a matrix factorisation algorithm that decomposes the user-item interaction matrix into latent factor vectors, enabling prediction of unobserved interactions.

Key parameters tuned via MLflow:
- `rank` — number of latent factors
- `maxIter` — training iterations
- `regParam` — L2 regularisation strength
- `alpha` — confidence scaling for implicit feedback

### 3. MLflow Experiment Tracking
- Logged all hyperparameter combinations as MLflow runs
- Tracked RMSE on validation set per run
- Selected best model based on minimum RMSE
- Registered final model in MLflow model registry

### 4. Evaluation
- Root Mean Squared Error (RMSE) on held-out test interactions
- Top-N game recommendations generated per user
- Cold-start strategy: `nan` handling configured for unseen users/items

## Key Findings

- ALS with `rank=10`, `regParam=0.1` produced lowest test RMSE in the evaluated grid
- MLflow experiment tracking enabled reproducible comparison across 12 hyperparameter combinations without manual record-keeping
- At scale (200K records), PySpark's distributed processing reduced training time significantly vs single-machine approaches

## How to Run

1. Import the `.html` notebook into Databricks (File → Import → HTML)
2. Attach to a Spark cluster (Community Edition is sufficient)
3. Upload the Steam-200K dataset to DBFS (`/FileStore/tables/`)
4. Run all cells

Dataset: [Steam-200K on Kaggle](https://www.kaggle.com/datasets/tamber/steam-video-games)

---

*MSc Data Science — Big Data & Cloud Computing module | University of Salford*
