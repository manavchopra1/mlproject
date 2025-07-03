
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

## 📬 Contact
- **Author:** Manav Chopra
- **Email:** [your-email@example.com]
- **LinkedIn:** [Your LinkedIn](https://www.linkedin.com/)

---

> *Happy Machine Learning!* 🎉
