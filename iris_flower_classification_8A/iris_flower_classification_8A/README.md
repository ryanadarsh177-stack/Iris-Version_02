# Iris Flower Classification (Section 8A)

An introductory supervised machine-learning project that classifies Iris flowers into
**setosa**, **versicolor**, or **virginica** using four measurements.

## Features

- Interactive Streamlit web app
- K-Nearest Neighbors (KNN) and Logistic Regression classifiers
- StandardScaler preprocessing through scikit-learn pipelines
- Train/test split with stratification
- Test accuracy comparison
- Per-species classification report and confusion matrix
- Live prediction from user-entered measurements
- Dataset included locally, so no dataset download is required at runtime

## Input features

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Project structure

```text
iris_flower_classification_8A/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── iris.csv
```

## Run locally

Python 3.10 or newer is recommended.

1. Extract the ZIP file.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (optional but recommended).

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

## Deploy on GitHub and Streamlit Community Cloud

1. Create a new GitHub repository, for example `iris-flower-classification`.
2. Extract this ZIP file and upload the **contents** of the project folder to the repository root.
3. Commit the files.
4. Open Streamlit Community Cloud and connect your GitHub account.
5. Create a new app, select the repository and branch, and set the main file path to `app.py`.
6. Deploy. The app reads `data/iris.csv` using a path relative to `app.py`.

## Dataset note

The CSV included in `data/iris.csv` is a local copy of the classic 150-row Iris dataset
distributed by scikit-learn, which is based on the well-known Fisher Iris dataset.
It contains the four standard measurements and the three species labels. It is included
to make the project reproducible and avoid a runtime dependency on Kaggle downloads.
If your instructor specifically requires a Kaggle-downloaded file, you can replace this
CSV with the equivalent Iris dataset after checking that the column names are mapped
to the names used by `app.py`.

## Methodology

1. Load the dataset.
2. Separate the four input measurements from the species label.
3. Split the dataset into 80% training and 20% test sets, stratified by species.
4. Scale the features and train KNN (5 neighbours) and Logistic Regression.
5. Evaluate both models on the same held-out test set using accuracy.
6. Use the selected trained model to classify the user's input measurements.

## Limitations

This is a small educational dataset. The accuracy shown is based on one train/test split,
so it should not be interpreted as a guarantee of performance on every real-world flower.
The probability shown by a classifier is a model estimate, not a guaranteed measure of
real-world certainty.

## Dependencies

See `requirements.txt`.
