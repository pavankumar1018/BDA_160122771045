# Big Data Analytics (22ADC12)
##  PySpark ML Project: Classification, Clustering, and Recommendation

##  Student Details
**Jammula Pavan Kumar**  
**160122771045**
**AD&DS-1(I1) SEM-6**

This project showcases three different machine learning models implemented using **Apache Spark (PySpark)**:

1. **Classification** using Logistic Regression (Titanic-style dataset)
2. **Clustering** using Gaussian Mixture Model (synthetic 2D data)
3. **Recommendation Engine** using ALS (book ratings)

---

##  Requirements

- Python 3.x
- Apache Spark (PySpark)
- Jupyter Notebook or a Python IDE (optional)

You can install PySpark using pip:

```bash
pip install pyspark
```

---

## Project Structure

```
/
│
├── Classification.py          # Titanic-style classification
├── Clustering.py              # Gaussian clustering on synthetic data
├── Recommendation System.py   # ALS-based book recommender
├── README.md                  # You're here!
```

---

##  Model Summaries

### 🔹 1. Classification (Titanic-style)
- **Algorithm**: Logistic Regression
- **Features**: Sex, Age, SibSp, Parch, Fare
- **Label**: Survived
- **Evaluation**: AUC (Binary Classification)

### 🔹 2. Clustering (GMM)
- **Algorithm**: Gaussian Mixture Model
- **Data**: Synthetic 2D points
- **Output**: Cluster centers & assignments

### 🔹 3. Recommendation (ALS)
- **Algorithm**: ALS (Alternating Least Squares)
- **Data**: Book ratings (userId, bookId, rating)
- **Output**: RMSE + Top-N recommendations for each user

---

##  How to Run

```bash
spark-submit Classification.py
spark-submit Clustering.py
spark-submit Recommendation System.py
```

Or run inside a Python script or Jupyter notebook if preferred.

---

##  Notes

- All datasets used are **synthetic** and embedded within the code for easy testing.
- The models can be extended to use real datasets (e.g., from CSV, databases, etc.).
- Evaluation metrics include **AUC**, **RMSE**, and **Silhouette Score** depending on the task.

---



---

##  License

This project is open-source and free to use under the MIT License.
