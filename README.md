# 🍽️ AI Food Waste Prediction & Smart Donation Recommendation System

<p align="center">

<img src="assets/banner.png" alt="Project Banner" width="100%">

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)

![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)

![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green?style=for-the-badge)

![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

</p>

---

# 📑 Table of Contents

- Project Overview
- Problem Statement
- Objectives
- Why This Project?
- Features
- Technologies Used
- Dataset
- Machine Learning Workflow
- Models Used
- Results
- Project Structure
- Installation
- Usage
- Streamlit Dashboard
- Future Improvements
- Author

---

# 📌 Project Overview

The **AI Food Waste Prediction & Smart Donation Recommendation System** is an end-to-end Machine Learning application developed to predict the amount of food waste generated in restaurants, hotels, catering services, and event management businesses.

The system analyzes operational factors such as:

- Type of Food
- Number of Guests
- Event Type
- Storage Conditions
- Purchase History
- Preparation Method
- Pricing
- Seasonality

Using these inputs, the machine learning model predicts the expected food waste and provides intelligent recommendations to minimize waste and improve food management.

---

# 🌍 Problem Statement

Food waste has become one of the biggest global challenges.

According to the United Nations Environment Programme (UNEP), millions of tons of food are wasted every year across restaurants, hotels, supermarkets, and households.

This leads to:

- Economic Loss
- Environmental Pollution
- Greenhouse Gas Emissions
- Increased Operational Costs
- Food Insecurity

Most restaurants estimate food demand manually, which often results in over-preparation and unnecessary waste.

This project addresses that problem by using Artificial Intelligence and Machine Learning to predict food waste before it occurs.

---

# 🎯 Objectives

The main objectives of this project are:

- Predict food waste using Machine Learning
- Help restaurants reduce operational costs
- Improve food preparation planning
- Reduce environmental impact
- Support smart food donation decisions
- Build an interactive decision support dashboard

---

# ⭐ Why This Project?

Unlike traditional prediction systems, this project provides:

✅ Machine Learning Prediction

✅ Smart Recommendation System

✅ Interactive Dashboard

✅ PDF Report Generation

✅ Streamlit Web Application

✅ End-to-End Deployment Ready

This project demonstrates the complete Machine Learning lifecycle from data preprocessing to deployment.

---

# 🚀 Key Features

✔ Food Waste Prediction

✔ Restaurant Decision Support

✔ AI Recommendation Engine

✔ Interactive Dashboard

✔ Data Visualization

✔ Model Comparison

✔ PDF Report Download

✔ Professional Streamlit UI

✔ GitHub Portfolio Ready

✔ Cloud Deployment Ready

---

# 🛠 Technologies Used

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-Learn |
| Models | Random Forest, Gradient Boosting, XGBoost |
| Deployment | Streamlit |
| Report Generation | FPDF |
| Version Control | Git & GitHub |

---

# 📊 Dataset Information

Dataset Name:

Food Wastage in Restaurants Dataset

Dataset includes operational information such as:

- Type of Food
- Number of Guests
- Event Type
- Quantity of Food
- Storage Conditions
- Purchase History
- Seasonality
- Preparation Method
- Geographical Location
- Pricing

Target Variable:

**Wastage Food Amount**
---

# 📂 Dataset Information

## Dataset Source

The project uses a **Restaurant Food Wastage Dataset** collected from restaurant operations.

The dataset contains historical information regarding food preparation, customer demand, purchasing behavior, and food wastage.

### Input Features

| Feature | Description |
|----------|-------------|
| Type of Food | Category of prepared food |
| Number of Guests | Expected number of customers |
| Event Type | Occasion of food preparation |
| Quantity of Food | Amount of food prepared |
| Storage Conditions | Food storage quality |
| Purchase History | Previous purchasing trend |
| Seasonality | Seasonal impact |
| Preparation Method | Cooking method |
| Geographical Location | Restaurant location |
| Pricing | Food price |
| Wastage Food Amount | Target variable |

---

# 📊 Exploratory Data Analysis (EDA)

Before training the models, Exploratory Data Analysis (EDA) was performed to understand the dataset.

The following analyses were conducted:

- Dataset Shape
- Missing Values
- Duplicate Records
- Data Types
- Statistical Summary
- Correlation Analysis
- Distribution Analysis
- Outlier Detection

### Visualizations

✔ Histogram

✔ Box Plot

✔ Correlation Heatmap

✔ Count Plot

✔ Scatter Plot

✔ Distribution Plot

These visualizations helped identify important patterns and relationships within the data.

---

# 🧹 Data Preprocessing

High-quality data preprocessing is essential for building an accurate machine learning model.

The following preprocessing steps were applied:

### 1. Handling Missing Values

- Checked missing values
- Removed or handled null records

### 2. Removing Duplicate Records

Duplicate observations were removed to improve model performance.

### 3. Label Encoding

Categorical variables were converted into numerical values using Label Encoding.

Encoded Features:

- Type of Food
- Event Type
- Storage Conditions
- Purchase History
- Seasonality
- Preparation Method
- Geographical Location

### 4. Feature Selection

The following independent variables were selected:

- Type of Food
- Number of Guests
- Event Type
- Quantity of Food
- Storage Conditions
- Purchase History
- Seasonality
- Preparation Method
- Geographical Location
- Pricing

Target Variable

- Wastage Food Amount

---

# 🤖 Machine Learning Pipeline

The complete machine learning workflow is illustrated below.

```text
Restaurant Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Label Encoding
        │
        ▼
Train-Test Split
        │
        ▼
Machine Learning Models
        │
        ▼
Performance Evaluation
        │
        ▼
Best Model Selection
        │
        ▼
Model Serialization (.pkl)
        │
        ▼
Streamlit Deployment
```

---

# 🌲 Machine Learning Models

Multiple regression algorithms were trained and compared.

## 1️⃣ Random Forest Regressor

Random Forest combines multiple decision trees to improve prediction accuracy.

### Advantages

- Handles non-linear relationships
- Reduces overfitting
- Robust to noisy data

---

## 2️⃣ Gradient Boosting Regressor

Gradient Boosting builds trees sequentially to minimize prediction error.

### Advantages

- High predictive accuracy
- Excellent performance on structured datasets

---

## 3️⃣ XGBoost Regressor

XGBoost is an optimized boosting algorithm widely used in industry and machine learning competitions.

### Advantages

- Fast training
- Excellent accuracy
- Handles missing values efficiently

---

### Advantages

- Minimal preprocessing
- High prediction performance
- Reduced overfitting

---

# 📈 Model Evaluation Metrics

The following regression metrics were used to evaluate model performance.

| Metric | Description |
|----------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² Score | Coefficient of Determination |

A higher R² Score and lower MAE/RMSE indicate better model performance.

---

# 🏆 Model Comparison

After training all models, their performances were compared using evaluation metrics.

The model with the highest **R² Score** was selected as the final model for deployment.

Example:

| Model | MAE | RMSE | R² Score |
|---------|------|-------|-----------|
| Random Forest | 1.638564 | 7.453367 | 0.928096 |
| Gradient Boosting | 2.341812 | 3.093666 | 0.907670 |
| XGBoost | 1.925293 | 2.873442 | 0.920347 |
---

# 💾 Model Saving

The best-performing model was serialized using **Joblib**.

```python
joblib.dump(model, "food_waste_prediction_model.pkl")
```
---

# 🌐 Streamlit Web Application

The project includes a fully interactive **Streamlit web application** that allows users to predict food waste without writing any code.

## Application Features

- 🏠 Professional Home Page
- 🍽️ Food Waste Prediction Form
- 🤖 AI-Based Recommendation System
- 📊 Interactive Dashboard
- 📈 Data Visualization
- 📄 PDF Report Generation
- 📱 Responsive User Interface

---

# 📊 Dashboard Features

The dashboard provides meaningful insights through interactive visualizations.

## Available Charts

- Food Waste Distribution
- Event Type Analysis
- Food Category Analysis
- Prediction Results
- Feature Importance
- Model Comparison

These charts help restaurant managers understand food waste patterns and make informed decisions.

---

# 📁 Project Structure

```text
AI-Food-Waste-Prediction-System/
│
├── app.py
├── Food_Waste_Prediction.ipynb
├── food_wastage_data.csv
├── food_waste_prediction_model.pkl
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── assets/
│   ├── banner.png
│   └── logo.png
│
├── screenshots/
│   ├── home.png
│   ├── prediction.png
│   ├── dashboard.png
│   └── report.png
│
└── reports/
    └── sample_report.pdf
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/farhan-ml/AI-Food-Waste-Prediction-System.git
```

Go to project folder

```bash
cd AI-Food-Waste-Prediction-System
```

Install required libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Launch the Streamlit application

```bash
streamlit run app.py
```

After running the command, open the local URL shown in the terminal, usually:

```
http://localhost:8501
```

---

# ☁️ Deployment

The project can be deployed on:

- Streamlit Community Cloud
- Render
- Hugging Face Spaces
- Azure App Service
- AWS EC2
- Google Cloud Platform

---

# 📸 Application Screenshots

## 🏠 Home Page

> Add screenshot here

```
screenshots/home.png
```

---

## 🍽️ Prediction Page

> Add screenshot here

```
screenshots/prediction.png
```

---

## 📊 Dashboard

> Add screenshot here

```
screenshots/dashboard.png
```

---

## 📄 PDF Report

> Add screenshot here

```
screenshots/report.png
```

---

# 📈 Expected Outcomes

Using this system, restaurants can:

- Reduce food waste
- Improve inventory planning
- Lower operational costs
- Support food donation initiatives
- Improve sustainability practices
- Make data-driven decisions

---

# 🚀 Future Improvements

Future versions of this project may include:

- Deep Learning Models
- Real-Time Prediction
- Cloud Database Integration
- Restaurant POS Integration
- IoT-Based Inventory Monitoring
- QR Code Reports
- Mobile Application
- Multi-language Support
- Email Notifications
- Live Analytics Dashboard

---

# 🎓 Learning Outcomes

This project demonstrates practical experience in:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Regression Modeling
- Model Evaluation
- Model Deployment
- Streamlit Development
- Data Visualization
- Git & GitHub
- End-to-End Machine Learning Projects

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Matplotlib | Visualization |
| Seaborn | Statistical Visualization |
| Plotly | Interactive Charts |
| Scikit-learn | Machine Learning |
| XGBoost | Regression Model |
| CatBoost | Regression Model |
| Streamlit | Web Application |
| Joblib | Model Serialization |
| FPDF2 | PDF Report Generation |
| Git | Version Control |
| GitHub | Repository Hosting |

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve this project:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Submit a Pull Request.

---

# 📜 License

This project is licensed under the **MIT License**.

See the LICENSE file for complete details.

---

# 👨‍💻 Author

**Muhammad Farhan**

**BS Information Technology**

📧 Email: fchandio717@gmail.com

🔗 GitHub: https://github.com/farhan-ml

🔗 LinkedIn: https://www.linkedin.com/in/muhammad-farhan

---

# ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

📢 Share it with others

---

# 🙏 Acknowledgements

Special thanks to:

- OpenAI
- Scikit-learn Community
- Streamlit Team
- XGBoost Developers
- CatBoost Developers
- Kaggle Community
- Python Community

---

<p align="center">

⭐ If you like this project, don't forget to Star the Repository ⭐

</p>
