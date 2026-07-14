import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Smart Lender - Loan Approval Prediction System\n",
    "## Exploratory Data Analysis and Modeling\n",
    "This notebook documents the data preprocessing and machine learning pipeline for predicting loan approvals using the official SkillWallet dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "from imblearn.over_sampling import SMOTE\n",
    "import joblib\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import os"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Official Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../dataset/loan_prediction.csv', sep='\\t')\n",
    "if len(df.columns) == 1:\n",
    "    df = pd.read_csv('../dataset/loan_prediction.csv', sep=',')\n",
    "df = df.drop('Loan_ID', axis=1, errors='ignore')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Exploratory Data Analysis (EDA)\n",
    "### 2.1 Distribution of Loan Status"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(6,4))\n",
    "sns.countplot(x='Loan_Status', data=df)\n",
    "plt.title('Loan Status Distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.2 Missing Values Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Missing Values:\\n\", df.isnull().sum())\n",
    "plt.figure(figsize=(10,6))\n",
    "sns.heatmap(df.isnull(), cbar=False, cmap='viridis')\n",
    "plt.title('Missing Values Heatmap')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.3 Distribution Plots for Numerical Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']\n",
    "df[numerical_cols].hist(bins=20, figsize=(12, 8))\n",
    "plt.suptitle('Numerical Features Distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing\n",
    "### 3.1 Handling Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Numerical: median\n",
    "for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:\n",
    "    if df[col].isnull().sum() > 0:\n",
    "        df[col].fillna(df[col].median(), inplace=True)\n",
    "\n",
    "# Categorical: mode\n",
    "for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:\n",
    "    if df[col].isnull().sum() > 0:\n",
    "        df[col].fillna(df[col].mode()[0], inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.2 Encoding Categorical Variables & Correlation Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_columns = [\n",
    "    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',\n",
    "    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',\n",
    "    'Credit_History', 'Property_Area'\n",
    "]\n",
    "\n",
    "encoders = {}\n",
    "categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']\n",
    "\n",
    "df_encoded = df.copy()\n",
    "for col in categorical_cols:\n",
    "    le = LabelEncoder()\n",
    "    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))\n",
    "    encoders[col] = le\n",
    "\n",
    "le_target = LabelEncoder()\n",
    "df_encoded['Loan_Status'] = le_target.fit_transform(df_encoded['Loan_Status'])\n",
    "\n",
    "X = df_encoded[feature_columns]\n",
    "y = df_encoded['Loan_Status']\n",
    "\n",
    "plt.figure(figsize=(10,8))\n",
    "sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=\".2f\")\n",
    "plt.title('Correlation Heatmap')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.3 Train/Test Split (Before Scaling to prevent Data Leakage)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n",
    "print(f'Original Training size: {X_train.shape}, Test size: {X_test.shape}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.4 Scaling and SMOTE (Applied ONLY to Training Data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "smote = SMOTE(random_state=42)\n",
    "X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)\n",
    "print(f'Balanced Training size: {X_train_bal.shape}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Model Training and Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'Decision Tree': DecisionTreeClassifier(random_state=42),\n",
    "    'Random Forest': RandomForestClassifier(random_state=42),\n",
    "    'KNN': KNeighborsClassifier(),\n",
    "    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\n",
    "}\n",
    "\n",
    "results = {}\n",
    "best_model = None\n",
    "best_accuracy = 0\n",
    "best_model_name = \"\"\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train_bal, y_train_bal)\n",
    "    y_train_pred = model.predict(X_train_bal)\n",
    "    y_test_pred = model.predict(X_test_scaled)\n",
    "    \n",
    "    train_acc = accuracy_score(y_train_bal, y_train_pred)\n",
    "    test_acc = accuracy_score(y_test, y_test_pred)\n",
    "    results[name] = test_acc\n",
    "    \n",
    "    print(f'--- {name} ---')\n",
    "    print(f'Train Accuracy: {train_acc:.4f}')\n",
    "    print(f'Test Accuracy: {test_acc:.4f}')\n",
    "    print('Classification Report (Test):')\n",
    "    print(classification_report(y_test, y_test_pred))\n",
    "    \n",
    "    cm = confusion_matrix(y_test, y_test_pred)\n",
    "    plt.figure(figsize=(4,3))\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
    "    plt.title(f'{name} Confusion Matrix')\n",
    "    plt.xlabel('Predicted')\n",
    "    plt.ylabel('Actual')\n",
    "    plt.show()\n",
    "    \n",
    "    if test_acc > best_accuracy:\n",
    "        best_accuracy = test_acc\n",
    "        best_model = model\n",
    "        best_model_name = name\n",
    "\n",
    "print(f'\\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.1 Accuracy Comparison Chart"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8,5))\n",
    "sns.barplot(x=list(results.keys()), y=list(results.values()))\n",
    "plt.title('Model Accuracy Comparison')\n",
    "plt.ylim(0, 1.0)\n",
    "for i, v in enumerate(results.values()):\n",
    "    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.2 Feature Importance (Best Model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if hasattr(best_model, 'feature_importances_'):\n",
    "    plt.figure(figsize=(10,6))\n",
    "    importances = best_model.feature_importances_\n",
    "    indices = np.argsort(importances)[::-1]\n",
    "    plt.title(f'Feature Importances ({best_model_name})')\n",
    "    sns.barplot(x=importances[indices], y=[feature_columns[i] for i in indices], palette='viridis')\n",
    "    plt.xlabel('Relative Importance')\n",
    "    plt.ylabel('Features')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print(f'{best_model_name} does not support feature_importances_ directly.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Export Models and Scaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.makedirs('../models', exist_ok=True)\n",
    "joblib.dump(best_model, '../models/best_model.pkl')\n",
    "joblib.dump(scaler, '../models/scaler.pkl')\n",
    "joblib.dump(encoders, '../models/encoders.pkl')\n",
    "joblib.dump(feature_columns, '../models/feature_columns.pkl')\n",
    "print(f'Artifacts saved successfully for {best_model_name}.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Conclusion\n",
    "The data was carefully preprocessed to avoid data leakage by splitting the dataset before applying scaling and SMOTE. Various models were evaluated using train and test accuracy, classification reports, and confusion matrices. The selected model was automatically chosen based on the best test accuracy. All necessary artifacts (`best_model.pkl`, `scaler.pkl`, `encoders.pkl`, `feature_columns.pkl`) were successfully exported for deployment."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('c:\\\\Users\\\\91630\\\\OneDrive\\\\Desktop\\\\SmartLender\\\\notebooks\\\\01_EDA_and_Modeling.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print('Notebook generated successfully.')
