# Credit Card Fraud Detection Using PySpark and Random Forest

## Project Overview

This project aims to build a fraud detection system for credit card transactions using Apache Spark and a Random Forest classifier. It processes transaction data using PySpark, performs extensive exploratory data analysis (EDA), and trains a machine learning model to classify whether a transaction is fraudulent or legitimate. The model's performance is evaluated using multiple metrics and visualized through ROC curves, confusion matrices, and learning curves.

---

## Technologies Used

- Apache Spark (PySpark)
- Python 3.x
- Scikit-learn (metrics)
- Matplotlib & Seaborn (visualizations)
- MLflow (model tracking)
- Pandas & NumPy (data manipulation)

---

## Dataset Features

The dataset includes the following columns:

| Column                        | Description                                                    |
|------------------------------|----------------------------------------------------------------|
| `distance_from_home`         | Distance from the user's home where the transaction occurred   |
| `distance_from_last_transaction` | Distance from the previous transaction                     |
| `ratio_to_median_purchase_price` | Ratio to typical purchase price                          |
| `repeat_retailer`            | Indicates if the transaction was with a known retailer         |
| `used_chip`                  | Whether a chip was used                                        |
| `used_pin_number`            | Whether a PIN was used                                         |
| `online_order`               | Whether the transaction was placed online                     |
| `fraud`                      | Target variable (0 = Legitimate, 1 = Fraudulent)              |

---

## Model: Random Forest

A Random Forest classifier was trained using Spark MLlib. The model was evaluated on both training and test sets to detect overfitting. The performance was tracked using MLflow and measured with the following metrics:

- AUC Score
- Accuracy
- Precision
- Recall
- F1 Score

---

## Results

### Final Model Metrics

| Metric     | Training Set | Test Set |
|------------|--------------|----------|
| AUC        | 0.9668       | 0.9666   |
| Accuracy   | 0.9388       | 0.9386   |
| Precision  | 0.9432       | 0.9429   |
| Recall     | 0.9388       | 0.9386   |
| F1 Score   | 0.9407       | 0.9404   |

These results demonstrate that the model generalizes well and shows no signs of overfitting.

---

## Learning Curve

A learning curve was plotted by training the model on increasing fractions of the dataset (from 10% to 100%). The AUC scores for both training and test sets were computed at each stage. The curve shows stable convergence and consistent generalization across different training sizes.

---

## Visualizations

- **Confusion Matrix**: Shows prediction distribution.
- **ROC Curve**: Illustrates classifier performance.
- **Feature Importance**: Highlights influential features.
- **Learning Curve**: Demonstrates model learning over data size.

All plots are saved and tracked using MLflow artifacts.

---

## Memory Optimization

To avoid memory cache warnings, Spark was configured with the following settings:

```python
spark = SparkSession.builder \
    .appName("Credit Card Fraud Detection") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()