# 📱 Customer Churn Segmentation & Revenue Optimization — MTN Bénin

![R](https://img.shields.io/badge/Language-R-276DC3?style=flat&logo=r)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Telecom%20Analytics-orange)

## 📌 Project Overview

This project applies **unsupervised machine learning and predictive analytics** to a simulated telecom customer dataset inspired by MTN Bénin's subscriber base. The goal is to identify high-risk churn segments, support retention strategies, and optimize revenue allocation using data-driven insights.

---

## 🎯 Business Objectives

- Identify customer groups at high risk of churn using clustering techniques
- Build predictive models to anticipate customer attrition
- Generate actionable business recommendations to reduce churn and increase lifetime value
- Support revenue optimization through risk-based customer segmentation

---

## 🗂️ Dataset Description

| Variable | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `tenure_months` | Duration of subscription (months) |
| `monthly_charge` | Monthly billing amount (XOF) |
| `data_usage_gb` | Average monthly data consumption |
| `call_minutes` | Average monthly call duration |
| `num_complaints` | Number of complaints filed |
| `payment_delay` | Number of late payments |
| `churn` | Binary target: 1 = churned, 0 = retained |

> ⚠️ *Data is simulated for analytical demonstration purposes.*

---

## 🔧 Methods & Tools

### Segmentation (Unsupervised)
- **K-Means Clustering** — Identification of 3 customer risk profiles
- **Elbow Method & Silhouette Score** — Optimal cluster selection
- **PCA** — Dimensionality reduction for visualization

### Predictive Modeling (Supervised)
- **Logistic Regression** — Baseline churn probability model
- **Random Forest** — Feature importance and improved accuracy
- **XGBoost** — Final optimized model

### Tools
```
R | tidyverse | caret | xgboost | ggplot2 | factoextra | cluster
```

---

## 📊 Key Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 78.4% | 0.81 |
| Random Forest | 84.1% | 0.88 |
| XGBoost | 86.7% | 0.91 |

### Customer Segments Identified

| Segment | Profile | Churn Risk |
|---|---|---|
| 🔴 Cluster 1 | High complaints, low usage, payment delays | **High** |
| 🟡 Cluster 2 | Medium tenure, average spend | **Medium** |
| 🟢 Cluster 3 | Long tenure, high usage, no complaints | **Low** |

---

## 📁 Project Structure

```
churn-segmentation-mtn/
│
├── data/
│   └── simulated_mtn_customers.csv
├── scripts/
│   ├── 01_data_preparation.R
│   ├── 02_clustering_analysis.R
│   ├── 03_predictive_modeling.R
│   └── 04_visualizations.R
├── outputs/
│   ├── cluster_plot.png
│   ├── roc_curves.png
│   └── feature_importance.png
└── README.md
```

---

## 💡 Business Recommendations

1. **Target Cluster 1** with proactive retention offers (discounts, loyalty programs)
2. **Deploy early warning system** using XGBoost scores to flag at-risk customers monthly
3. **Reduce complaint resolution time** — top predictor of churn in all models
4. **Upsell data packages** to Cluster 2 to increase stickiness before churn window

---

## 👤 Author

**Adededji Djamiou ABAYOMI**
Data Analyst | Quantitative Modeling | Business Intelligence
📍 Montréal, QC, Canada
📧 abayomi.adededji.djamiou@gmail.com
🔗 [LinkedIn](https://linkedin.com)
