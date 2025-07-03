# 📊 MLProject: Student Performance Indicator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-Model-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A robust, end-to-end machine learning pipeline to predict student performance based on demographic and educational features. This project demonstrates best practices in data ingestion, transformation, model training, experiment tracking, and reproducibility.

---

## 📚 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data & Artifacts](#data--artifacts)
- [Experiment Tracking](#experiment-tracking)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Features
- **Data Ingestion**: Pulls data from a MySQL database and splits into train/test sets.
- **Data Transformation**: Handles missing values, encodes categorical features, and scales numerical features.
- **Model Training**: Trains and evaluates multiple regression models (CatBoost, XGBoost, Random Forest, etc.) with hyperparameter tuning.
- **Experiment Tracking**: Uses MLflow for experiment logging and model registry.
- **Reproducibility**: All steps are modular and tracked; artifacts are versioned.
- **Notebooks**: EDA and model training notebooks for exploration and demonstration.

---

## 🗂 Project Structure
```
mlproject/
│   app.py                # Main pipeline runner
│   main.py               # (Reserved for CLI/API entry)
│   requirements.txt      # Python dependencies
│   Dockerfile            # (Optional) Docker support
│   setup.py              # (Optional) Install as package
│
├── artifacts/            # Data & model artifacts
│   ├── train.csv, test.csv, model.pkl, preprocessor.pkl
│
├── src/mlproject/
│   ├── components/       # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_monitering.py
│   ├── pipelines/        # (Reserved for pipeline scripts)
│   ├── utils.py, logger.py, exception.py
│
├── notebook/             # Jupyter notebooks (EDA, training)
│   ├── EDASP.ipynb
│   └── 2. MODEL TRAINING.ipynb
└── ...
```

---

## ⚙️ Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd mlproject
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or, for development:
   ```bash
   pip install -e .
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the root directory with your MySQL credentials:
     ```ini
     host=YOUR_HOST
     user=YOUR_USER
     password=YOUR_PASSWORD
     database=YOUR_DATABASE
     ```

---

## ▶️ Usage
Run the main pipeline (data ingestion, transformation, training):
```bash
python app.py
```

- The pipeline will:
  - Ingest data from MySQL
  - Split and save train/test sets
  - Transform data and save preprocessor
  - Train and evaluate models, saving the best model
  - Log metrics and models to MLflow

**Artifacts** will be saved in the `artifacts/` directory.

---

## 📦 Data & Artifacts
- **artifacts/train.csv, test.csv**: Training and test datasets
- **artifacts/model.pkl**: Trained model
- **artifacts/preprocessor.pkl**: Data preprocessor
- **artifacts/raw.csv.dvc**: DVC file for data versioning

---

## 📈 Experiment Tracking
- **MLflow** is used for experiment tracking and model registry.
- Run the MLflow UI:
  ```bash
  mlflow ui
  ```
- Track runs, metrics, and models in the `mlruns/` directory.

---

## 📒 Notebooks
- **notebook/EDASP.ipynb**: Exploratory Data Analysis (EDA)
- **notebook/2. MODEL TRAINING.ipynb**: Model training and evaluation

---

## 🤝 Contributing
Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## 📝 License
This project is licensed under the MIT License.

---

> *Happy Machine Learning!* 🎉
