
# 🔍 Employee Layoff Prediction

📊 **Predict Employee Attrition with Machine Learning**  
A robust machine learning system that predicts employee layoffs using various classification models. Designed for HR analytics, this project uses historical employee data to identify potential attrition risks and supports proactive workforce management.

📅 *Last Updated: July 12, 2025, 11:45 PM IST*

---

## 📌 Overview

This project leverages structured HR datasets and machine learning algorithms to forecast employee attrition. The system is built using Python and applies modern classification techniques such as Random Forest, AdaBoost, SVM, and others. It includes data preprocessing, model evaluation, SMOTE-based oversampling, and PCA for dimensionality reduction.

💡 Best Use Case: Helping HR departments predict and reduce employee turnover with actionable data insights.

---

## ✅ Features

- ⚙️ Multiple ML models implemented: Logistic Regression, Random Forest, SVM, AdaBoost, KNN, Decision Tree, Naive Bayes.
- 🧠 Predictive insights into employee attrition risk.
- 🧪 SMOTE applied to handle imbalanced datasets.
- 📉 PCA support for dimensionality reduction and better visualization.
- 📊 Feature importance ranking (e.g., MonthlyIncome, OverTime, Age).
- 🧼 Full pipeline: Data cleaning, encoding, scaling, tuning, and evaluation.
- 📈 Model performance comparisons using Accuracy, Precision, Recall, F1-score.

---

## 🧠 Technologies Used

- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook

---

## 📂 Project Structure

```

employee-layoff-prediction/
├── data/
│   └── employee\_data.csv
├── models/
│   └── best\_model\_rf.pkl
├── notebooks/
│   └── RandomForest With Oversampling.ipynb
├── utils/
│   └── preprocessing.py
├── main.py
└── README.md

````

---

## 🧪 Model Performance Summary

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 75.3%    | 75.5%     | 75.1%  | 75.2%    |
| Decision Tree      | 87.2%    | 82.6%     | 94.4%  | 88.0%    |
| Random Forest 🏆    | **99.2%**| **98.6%** | **99.8%** | **99.2%** |
| AdaBoost           | 98.9%    | 98.2%     | 99.8%  | 99.0%    |
| SVM                | 95.1%    | 91.2%     | 99.8%  | 95.3%    |
| KNN                | 89.4%    | 82.6%     | 99.8%  | 90.4%    |

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Steps

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/employee-layoff-prediction.git
cd employee-layoff-prediction
````

2. **Create a Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/macOS
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. **Run the Notebook**

```bash
jupyter notebook notebooks/RandomForest With Oversampling.ipynb
```

2. **Or Run the Python Script**

```bash
python main.py
```

3. **Model Output**

* Generates classification report.
* Saves model to `models/best_model_rf.pkl`.

---

## 📈 Future Enhancements

* 📡 Real-time prediction dashboard (Streamlit or Dash)
* 🧠 Deep Learning with LSTM or ANN for time-series employee behavior
* 🛡️ Fairness-aware ML models and SHAP explanations
* 🔌 Integration with HRM/ERP systems for automated monitoring

---

## 👨‍💻 Contributors

* Pragy Upadhyay

  👨‍🏫 *Under the guidance of Dr. Bajrang Bansal*
  🧑‍🏫 Jaypee Institute of Information Technology, Noida

---

## 🤝 Contributing

Contributions are welcome!
If you'd like to improve or extend this project:

1. Fork the repo
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to your branch (`git push origin feature-name`)
5. Submit a pull request

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

For queries or suggestions, open an [issue](https://github.com/your-username/employee-layoff-prediction/issues) or email us at pragy34@gmail.com
---

🔮 **Proactively reduce employee churn — one prediction at a time!**

````

---

### ✅ Next Steps for GitHub

- Replace `your-username` and `your-email@example.com` with your GitHub username and contact.
- Add a `LICENSE` file (e.g., MIT License).
- Create a `requirements.txt`:
  ```txt
  pandas
  numpy
  scikit-learn
  imbalanced-learn
  matplotlib
  seaborn
  jupyter
````

Would you like me to generate a `main.py` version of the notebook too?
