# 📊 Employee Attrition Analysis Dashboard

### Interactive HR Analytics & Machine Learning Dashboard

The **Employee Attrition Analysis Dashboard** is a data-driven HR analytics project designed to analyze employee resignation patterns, identify workforce trends, assess attrition risk, and support better HR decision-making.

The application combines **Python, Streamlit, Plotly, Pandas, and Scikit-learn** to transform HR data into interactive visualizations, predictive insights, and strategic recommendations.

The dashboard also provides **Arabic / RTL support**, making the analysis accessible through a localized HR interface.

---

## ✨ Key Features

- 📊 Interactive HR analytics dashboard
- 📈 Employee attrition trend analysis
- 👥 Gender-based workforce analysis
- 🏢 Department-level attrition analysis
- ⏳ Employee tenure and service-duration analysis
- 🎯 Resignation reason analysis
- ⚠️ Employee attrition risk assessment
- 🤖 Machine learning prediction using Random Forest
- 📌 Feature importance analysis
- 🔮 Future attrition forecasting
- 💡 Strategic HR recommendations
- 💰 Retention and ROI analysis
- 📤 CSV data and report export
- 🌐 Arabic / RTL user interface
- 🧹 Multi-year Excel data preprocessing

---

## 🤖 Machine Learning

The project includes a predictive employee attrition model built using **Random Forest Classification**.

The model analyzes factors such as:

- Employee age
- Department
- Job title
- Gender
- Length of service
- Performance rating
- Salary grade

The dashboard provides:

- Model accuracy
- Feature importance
- Attrition risk indicators
- Predictive workforce insights
- Future attrition forecasting

---

## 📊 Analytics Modules

The dashboard explores employee attrition from several perspectives:

### Attrition Trends

Analyze resignation patterns across multiple years and identify changes over time.

### Department Analysis

Identify departments with higher employee turnover and compare attrition levels across the organization.

### Resignation Reasons

Analyze the most common reasons employees leave and identify the main drivers of attrition.

### Employee Demographics

Explore resignation patterns based on:

- Gender
- Age
- Age group
- Job title
- Department

### Tenure Analysis

Examine how employee service duration relates to resignation behavior.

### Risk Analysis

Employees are evaluated using several risk indicators to highlight patterns associated with higher attrition risk.

### Strategic Recommendations

The system generates HR-focused recommendations to support employee retention and workforce planning.

---

## 🛠️ Tech Stack

### Programming & Data Analysis

- Python
- Pandas
- NumPy

### Dashboard & Visualization

- Streamlit
- Plotly
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- Random Forest Classifier
- Label Encoding
- Train/Test Split
- Model Accuracy Evaluation
- Feature Importance

### Data Processing

- OpenPyXL
- XLrd
- Excel data preprocessing
- Multi-year dataset integration

---

## 🗂️ Project Structure

```text
employee-attrition-analysis-dashboard/
│
├── hr_dashboard.py
│   └── Main Streamlit HR analytics and ML dashboard
│
├── HR dashboard.py
│   └── Additional dashboard implementation
│
├── hr.py
│   └── HR attrition analytics implementation
│
├── import and preprocessing.py
│   └── Multi-year Excel loading, cleaning and preprocessing
│
├── config.json
│   └── Dashboard and model configuration
│
├── style.html
│   └── Additional interface styling
│
├── deployment_guide.md
│   └── Deployment and usage instructions
│
├── requirements.txt
│   └── Python dependencies
│
└── .gitignore
    └── Excludes private HR datasets and local files
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have:

- Python 3.10+
- pip
- Git

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Aya-Ghazii/employee-attrition-analysis-dashboard.git

cd employee-attrition-analysis-dashboard
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Dashboard

```bash
streamlit run hr_dashboard.py
```

Streamlit should automatically open the application in your browser.

Default local address:

```text
http://localhost:8501
```

---

## 🧹 Data Preprocessing

The project includes a dedicated preprocessing script:

```text
import and preprocessing.py
```

It supports:

- Loading multiple yearly Excel files
- Combining datasets
- Cleaning column names
- Handling missing values
- Removing duplicate records
- Optimizing data types
- Detecting and processing outliers
- Exporting processed datasets

---

## 🔐 Data Privacy

The original HR Excel datasets are **not included in this public GitHub repository**.

They are excluded through `.gitignore` because employee-level HR information should not be publicly exposed.

The main dashboard can demonstrate its analytics using generated/sample data, while the preprocessing pipeline can be used locally with authorized HR datasets.

---

## 🌍 Arabic Support

The dashboard includes Arabic-language and RTL interface support for HR analytics.

Features include:

- Arabic dashboard labels
- Arabic resignation reasons
- Arabic department classifications
- RTL page layout
- Arabic-friendly visualizations and reporting

---

## 🎯 Project Purpose

This project demonstrates how **Data Science and Machine Learning** can support Human Resources teams by transforming workforce data into actionable insights.

It showcases skills in:

`Python` · `Data Analysis` · `Machine Learning` · `Streamlit` · `Plotly` · `Pandas` · `Scikit-learn` · `Data Visualization` · `Data Preprocessing` · `HR Analytics`

---

## 🔮 Future Improvements

Potential enhancements include:

- Employee-level real-time attrition prediction
- Additional classification model comparison
- Precision, Recall, F1-score and ROC-AUC evaluation
- SHAP-based model explainability
- Database integration
- Automated HR alerts
- Interactive employee risk profiles
- Cloud deployment
- Advanced workforce forecasting

---

## 👩‍💻 Author

**Aya Ghazi**

Data Science & Computer Science

GitHub: [Aya-Ghazii](https://github.com/Aya-Ghazii)

---

## 📌 Project Status

✅ Interactive HR Dashboard  
✅ Attrition Analysis  
✅ Machine Learning Model  
✅ Risk Assessment  
✅ Future Forecasting  
✅ Strategic Recommendations  
✅ Arabic / RTL Support  
✅ Data Privacy Protection  

---

⭐ If you found this project interesting, feel free to star the repository.
