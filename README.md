# Loan Approval Prediction API & Frontend

This project predicts **loan approval status (Approved / Rejected)** using machine learning.  
It provides:  
- A **FastAPI backend** for serving predictions via REST API.  
- A **Streamlit frontend** that interacts with the API.  
- A trained **RandomForestClassifier model** (best-performing model).  
- Full pipeline including **preprocessing, SMOTE handling, and model comparision**.  

Dataset used: [Loan Approval Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data)  

---

## Features

- Preprocessing pipeline with:
  - Handling cheacking missing values, outlier removal, and correlation analysis.
  - Scaling (`RobustScaler`) for numerical features.
  - Encoding (`OrdinalEncoder` + `OneHotEncoder`) for categorical features.
  - SMOTENC oversampling for class imbalance.

- Models compared:
  - Logistic Regression → **92% accuracy**  
  - Decision Tree → **98% accuracy**  
  - Naive Bayes → **94% accuracy**  
  - Random Forest → **98% accuracy (selected)**  

- Feature importance (top contributors):  
  - **CIBIL score (0.83)**  
  - Loan term, Loan amount, Asset values, Dependents, Education, Self-employed.  

---

## Project Structure

```
LOAN_APPROVAL/
│── api.py              # FastAPI backend for predictions
│── frontend.py         # Streamlit app consuming the API
│── constants.py        # Paths & constants
│── loan_approval.ipynb # Model training & evaluation notebook
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│
├── data/               # Dataset file
├── loan/               # Pydantic schemas & request handling
├── models/             # Trained ML models (.joblib)
└── __pycache__/        # Cache files
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the FastAPI Backend
```bash
uvicorn api:app --reload
```
- Default: `Your LocalHost`  
- API endpoint: `POST /api/loan/v1/predict`

Sample request:
```json
{
  "no_of_dependents": 1,
  "education": "Graduate",
  "selfEmployed": "No",
  "incomeAnnum": 600000,
  "loanAmount": 1200000,
  "loanTerm": 12,
  "cibilScore": 720,
  "residentialAssetsValue": 500000,
  "commercialAssetsValue": 200000,
  "luxuryAssetsValue": 300000,
  "bankAssetValue": 100000
}
```

Sample response:
```json
{
  "approved": "Approved",
  "probability": 0.98
}
```

---

### Run the Streamlit Frontend
```bash
streamlit run frontend.py
```
- Opens a local web UI where users can enter loan details.  
- Submits request to the FastAPI backend and displays prediction.

---

## Results

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 92%      | 0.95/0.87 | 0.92/0.92 | 0.94/0.90 |
| Decision Tree       | 98%      | 0.98/0.98 | 0.99/0.97 | 0.98/0.97 |
| Naive Bayes         | 94%      | 0.98/0.89 | 0.93/0.97 | 0.95/0.93 |
| Random Forest       | **98%**  | 0.98/0.97 | 0.98/0.97 | 0.98/0.97 |

- **Best Model: Random Forest (98% accuracy)**  

---

## Requirements

- Python 3.8+  
- Dependencies listed in `requirements.txt`

---

## License

This project is licensed under the MIT License.  
