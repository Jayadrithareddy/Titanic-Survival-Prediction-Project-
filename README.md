# Titanic-Survival-Prediction-Project-
The Titanic Survival Prediction project applies machine learning to forecast passenger survival using features like age, gender, and class. It involves data preprocessing, exploratory analysis, feature engineering, and building models such as Logistic Regression and Random Forest to evaluate survival patterns.
# 🚢 Titanic Survival Prediction

This project predicts the survival of passengers aboard the Titanic using **machine learning models**. It is based on the famous Kaggle Titanic dataset and demonstrates data preprocessing, feature engineering, exploratory analysis, and predictive modeling.

---

## 📌 Project Overview
The goal is to build a model that predicts whether a passenger survived the Titanic disaster based on features such as:
- Age  
- Gender  
- Passenger class  
- Family size  
- Ticket information  

---

## ⚙️ Steps Involved
1. **Data Preprocessing**  
   - Handle missing values (Age, Cabin, Embarked)  
   - Encode categorical features (Sex, Embarked)  
   - Feature scaling/normalization  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize survival rates by gender, class, age groups  
   - Correlation heatmaps and feature importance analysis  

3. **Feature Engineering**  
   - Create new features like FamilySize, Title, and IsAlone  
   - Drop irrelevant features  

4. **Model Building**  
   - Logistic Regression  
   - Decision Trees  
   - Random Forests  
   - Support Vector Machine (SVM)  
   - Gradient Boosting  

5. **Model Evaluation**  
   - Accuracy, Precision, Recall, F1-Score  
   - Cross-validation  

---

## 📊 Results
- Random Forest and Gradient Boosting provided the best accuracy.  
- Gender, class, and family size were strong indicators of survival.  

---

## 🛠️ Tech Stack
- **Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  

---

## 📂 Project Structure
├── data
│ ├── train.csv
│ └── test.csv
├── notebooks
│ └── Titanic_EDA_Model.ipynb
├── src
│ └── model.py
├── README.md
└── requirements.txt


---

## 🚀 How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction


Install dependencies

pip install -r requirements.txt


Run Jupyter Notebook or Python scripts to train and evaluate models.

📌 Future Work

Deploy model with Flask/Django

Add hyperparameter tuning (GridSearchCV, RandomizedSearchCV)

Improve feature engineering with external datasets 

Email -  settypallijayadrithareddy@gmail.com
Linkedin - https://www.linkedin.com/in/jayadrithareddy
