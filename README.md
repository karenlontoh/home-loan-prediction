# **Home Loan Approval Prediction 🏠**

Create a home loan approval predictor using machine learning techniques to streamline the approval process and improve customer satisfaction, and deploy it through Hugging Face.

## **Introduction 🏡**

Dream Housing Finance is a leading company offering home loans to prospective buyers. Many individuals face challenges with the lengthy approval process for home loans, often taking anywhere from 18 to 40 days, which can create anxiety among applicants. To address this, Dream Housing Finance aims to streamline the approval process by developing a predictive model to assess loan eligibility faster and more efficiently. This project involves analyzing a dataset of loan applications to predict whether a loan will be approved or not.

## **Data Overview 📊**

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset) and consists of 12 features aimed at determining the eligibility of home loans. The dataset provides essential information regarding applicants' personal and financial details, which can be used to predict whether their loan application will be approved.

Here’s a summary of the columns in the dataset:

| **Key Name**          | **Description**                                                                |
|-----------------------|--------------------------------------------------------------------------------|
| Loan_ID               | Unique Loan ID                                                                 |
| Gender                | Applicant gender (Male / Female)                                               |
| Married               | Applicant marital status (Y / N)                                               |
| Dependents            | Number of dependents (0 / 1 / 2 / 3+)                                          |
| Education             | Applicant education level (Graduate / Not Graduate)                            |
| Self_Employed         | Whether the applicant is self-employed (Y / N)                                 |
| ApplicantIncome       | Applicant income                                                               |
| CoapplicantIncome     | Coapplicant income                                                             |
| LoanAmount            | Amount requested for the loan                                                  |
| Loan_Amount_Term      | Term of the loan in months                                                     |
| Credit_History        | Customer past credit history (1: Good credit / 0: Bad credit)                  |
| Property_Area         | Urban / Semiurban / Rural                                                      |
| Loan_Status           | Loan approval status (Y / N)                                                  |

The dataset contains a total of **12 features** categorized into **8 categorical features** and **4 numerical features**. The **target variable** for prediction is **Loan_Status**, which indicates whether a loan is approved (Y) or rejected (N).

## **Background 🌍**

A **J.D. Power** survey found that **58% of customers** cited long loan approval times as a major factor in mortgage satisfaction. This extended approval process significantly affects the overall customer experience, with many prospective borrowers seeking quicker resolutions. To address these concerns, **Dream Housing Finance** aims to streamline the loan approval process through a predictive model that will expedite decision-making and improve customer satisfaction.

## **Methodology 🔍**

1. **Data Loading and Inspection**: Load the dataset and inspect key details like column names, data types, and value ranges to ensure data quality and readiness for analysis.
2. **Exploratory Data Analysis (EDA)**: Analyze the dataset to uncover patterns, correlations, and trends that can inform the model.
3. **Feature Engineering**: Enhance and prepare the dataset by transforming variables, handling missing values, and creating new features.
4. **Model Building**: Utilize machine learning algorithms to train models and predict loan approval status.
5. **Model Evaluation**: Assess the model performance using key evaluation metrics, with a focus on the F1-score, as it balances precision and recall, reducing both false positives and false negatives.
6. **Model Optimization and Deployment**: Fine-tune the best model and deploy it for reliable loan approval predictions.

## **Machine Learning Models Employed:**

- K-Nearest Neighbours (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost**

## **Model Analysis 🧮**

### Strength: High Performance
- **F1 Score**: Achieves a solid F1 Score of **0.88** on the test set and a **mean of 0.91** from cross-validation, demonstrating strong performance.
- **Consistency**: The low **standard deviation (0.03)** highlights reliable results across different data splits.

### Weakness: Performance Variability
- **F1 Score Range**: The F1 Scores range from **0.87 to 0.95**, indicating variability in performance, which could point to **overfitting** or the model being sensitive to specific data splits.

### Improvement: Enhance Robustness
- **Hyperparameter Tuning**: Further optimization of hyperparameters through **grid search** or **random search** can help reduce variability.
- **Training Data Diversification**: Incorporating more diverse or balanced data can enhance model generalization and minimize overfitting.

## **Conclusion 📈**

Dream Housing Finance has developed a predictive model that performs well but can be further refined. By optimizing the model's parameters and enhancing data diversity, the model can provide even more accurate predictions and faster loan approvals. This will help address the customer concerns about long processing times and ultimately lead to better customer satisfaction. Improving this model can streamline the approval process, reduce processing times, and increase overall operational efficiency.

## **Link to Model Deployment 🚀**

You can explore the deployed model and interact with it here:  
[Home Loan Prediction Model on Hugging Face](https://huggingface.co/spaces/karenlontoh/home-loan-prediction)

## **Libraries Used 🛠️**

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit

## **Author 👩‍💻**

Karen Lontoh  
LinkedIn: [Karmenia Ditabaya Lontoh](https://www.linkedin.com/in/karmenia-lontoh)
