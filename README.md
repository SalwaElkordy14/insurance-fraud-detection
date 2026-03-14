# insurance-fraud-detection
# 🛡️ Vehicle Insurance Fraud Detection

A Machine Learning project that detects fraudulent insurance claims.

## 📊 Dataset
- Source: Kaggle
- 30,000 records
- Binary Classification (Fraud / Legitimate)

## 🔧 Technologies Used
- Python
- Scikit-learn
- FastAPI
- Streamlit
- Pandas & NumPy

## 🤖 Models Trained
- Logistic Regression ✅ (Best Model)
- Naive Bayes
- KNN
- SVM
- Decision Tree
- Random Forest
- Gradient Boosting

## 🚀 How to Run

### Backend
```bash
uvicorn app:app --reload
```

### Frontend
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure
- `app.py` - FastAPI Backend
- `streamlit_app.py` - Streamlit Frontend
- `fraud_model.pkl` - Saved Model
