# Loan Approval Prediction

## 📌 Project Overview
This project predicts whether a loan application will be approved or rejected based on various applicant details. The dataset includes features such as income, loan amount, credit score, and employment status. A deep learning model is trained using TensorFlow/Keras to classify applications into **Approved** or **Rejected**.

## 🛠️ Technologies Used
- **Python** (pandas, NumPy, TensorFlow, scikit-learn)
- **Machine Learning** (Logistic Regression, Neural Networks)
- **Deep Learning** (Keras Sequential Model)
- **Data Preprocessing** (Feature Encoding, Normalization, Scaling)
- **Jupyter Notebook** for experimentation

## 📂 Repository Structure
```
Loan-Approval-Prediction/
│── data/
│   ├── loan_approval.csv           # Original dataset
│   ├── processed_data.csv          # Cleaned dataset
│── notebooks/
│   ├── data_preprocessing.ipynb    # Data cleaning & preprocessing
│   ├── model_training.ipynb        # Model training & evaluation
│   ├── model_testing.ipynb         # Model testing & validation
│── models/
│   ├── loan_approval_model.h5      # Saved trained model
│   ├── scaler.pkl                  # StandardScaler for preprocessing
│── scripts/
│   ├── train_model.py              # Script for model training
│   ├── predict.py                  # Script for making predictions
│── README.md                       # Project documentation
│── requirements.txt                 # Required Python libraries
│── LICENSE                         # Open-source license
│── .gitignore                       # Ignore unnecessary files
```

## 📊 Dataset Description
The dataset consists of financial and personal details of applicants.
| Column | Description |
|---------|----------------------|
| `no_of_dependents` | Number of dependents |
| `income_annum` | Annual income of applicant |
| `loan_amount` | Loan amount requested |
| `loan_term` | Loan repayment term (months) |
| `cibil_score` | Credit score of applicant |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value` | Value of commercial assets |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Total assets in bank |
| `education_ Not Graduate` | Education status (binary) |
| `self_employed_ Yes` | Employment type (binary) |
| `loan_status_ Rejected` | Loan rejection status (target) |

## 🏗️ Data Preprocessing
- **Remove leading/trailing spaces in column names**
- **Convert categorical variables using One-Hot Encoding**
- **Scale numeric features using `StandardScaler`**

## 🧠 Model Architecture
```python
model = Sequential([
    Dense(32, activation='relu', input_shape=(xtrain.shape[1],)),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

## 📈 Model Training
```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=20, batch_size=16, validation_data=(xtest, ytest))
```

## 🧪 Model Testing
```python
loss, accuracy = model.evaluate(xtest, ytest)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
```

## 🔮 Making Predictions
```python
# New Applicant Data
new_data = np.array([[2, 4100000, 12200000, 8, 417, 2700000, 2200000, 8800000, 3300000, 1, 1]])
new_data_df = pd.DataFrame(new_data, columns=feature_columns)
new_data_scaled = scaler.transform(new_data_df)

# Predict Approval/Rejection
prediction = model.predict(new_data_scaled)
predicted_class = (prediction > 0.5).astype(int)[0][0]

if predicted_class == 1:
    print("Loan Approved ✅")
else:
    print("Loan Rejected ❌")
```

## 📌 Results
The model predicts whether a loan application will be approved or rejected based on the provided applicant details.

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## 📬 Contact
- **GitHub**: [AllanOtieno254](https://github.com/AllanOtieno254)
- **LinkedIn**: [Allan Otieno Akumu](https://www.linkedin.com/in/allanotienoakumu)

---
🔹 **Contributions are welcome!** Feel free to fork this repository and improve the model. 🚀

