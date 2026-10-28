import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Iris Flower Classification" \
    "Made By YuvRaj And Shweta",
    page_icon="🌸",
    layout="wide"
)

DATA_PATH = Path(__file__).parent / "data" / "iris.csv"
FEATURES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
DISPLAY_FEATURES = {
    "sepal_length_cm": "Sepal length (cm)",
    "sepal_width_cm": "Sepal width (cm)",
    "petal_length_cm": "Petal length (cm)",
    "petal_width_cm": "Petal width (cm)",
}
SPECIES = ["setosa", "versicolor", "virginica"]

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["species"])
    if not required.issubset(data.columns):
        raise ValueError("The Iris CSV is missing one or more required columns.")
    return data

@st.cache_resource
def train_models():
    data = load_data()
    X = data[FEATURES]
    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "K-Nearest Neighbors (KNN)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, predictions),
            "predictions": predictions,
            "y_test": y_test,
            "report": classification_report(
                y_test, predictions, labels=SPECIES, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=SPECIES),
        }
    return data, models, results

st.title("🌸 Iris Flower Classification")
st.caption("Section 8A • Introductory supervised machine learning project")

st.markdown("""
This app predicts the species of an Iris flower from four measurements.
It trains and compares **K-Nearest Neighbors (KNN)** and **Logistic Regression**
using the same train/test split, then reports **accuracy** on the held-out test set.
""")

try:
    data, models, results = train_models()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Prediction settings")
    chosen_model_name = st.selectbox("Choose a model", list(models.keys()))
    st.write("Enter the flower's measurements in centimetres.")
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)

input_values = pd.DataFrame([{
    "sepal_length_cm": sepal_length,
    "sepal_width_cm": sepal_width,
    "petal_length_cm": petal_length,
    "petal_width_cm": petal_width,
}])

selected_model = models[chosen_model_name]
predicted_species = selected_model.predict(input_values)[0]
probabilities = selected_model.predict_proba(input_values)[0]
class_names = selected_model.classes_
confidence = float(np.max(probabilities))

st.subheader("Flower prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted species", predicted_species.title())
col2.metric("Model", "KNN" if chosen_model_name.startswith("K-Nearest") else "Logistic Regression")
col3.metric("Model confidence", f"{confidence:.1%}")

st.write("**Input measurements**")
pretty_input = input_values.rename(columns=DISPLAY_FEATURES)
st.dataframe(pretty_input, use_container_width=True, hide_index=True)

prob_df = pd.DataFrame({
    "Species": class_names,
    "Predicted probability": probabilities,
})
st.write("**Predicted probability by species**")
st.bar_chart(prob_df.set_index("Species"))

st.divider()
st.subheader("Model comparison")
comparison = pd.DataFrame([
    {"Model": name, "Test accuracy": result["accuracy"]}
    for name, result in results.items()
]).sort_values("Test accuracy", ascending=False)
metric_cols = st.columns(len(comparison))
for col, (_, row) in zip(metric_cols, comparison.iterrows()):
    col.metric(row["Model"], f"{row['Test accuracy']:.1%}")
st.dataframe(comparison, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.markdown("### Confusion matrix")
    result = results[chosen_model_name]
    cm = result["confusion_matrix"]
    fig, ax = plt.subplots()
    image = ax.imshow(cm)
    ax.set_xticks(range(len(SPECIES)), [s.title() for s in SPECIES], rotation=25)
    ax.set_yticks(range(len(SPECIES)), [s.title() for s in SPECIES])
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("Actual species")
    ax.set_title(chosen_model_name)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("### Classification report")
    report = pd.DataFrame(result["report"]).T
    st.dataframe(report.round(3), use_container_width=True)

st.divider()
st.subheader("About the machine-learning methods")
with st.expander("K-Nearest Neighbors (KNN)"):
    st.write("""
    KNN compares a new flower's measurements with flowers in the training data.
    It finds the five closest examples and predicts the species that appears most
    often among those neighbours. The measurements are scaled so features with
    larger numeric ranges do not dominate distance calculations.
    """)
with st.expander("Logistic Regression"):
    st.write("""
    Logistic Regression learns relationships between the four measurements and
    the three species. It estimates a probability for each species and chooses
    the species with the highest estimated probability. Feature scaling is
    applied before training.
    """)
with st.expander("Dataset and evaluation"):
    st.write(f"The included dataset contains **{len(data)} rows** and three species.")
    st.write("""
    The data is split into 80% training data and 20% test data. Stratification
    keeps the species proportions similar in both sets. Accuracy is the number
    of correct test predictions divided by the total number of test examples.
    Results can vary if the split or model settings change.
    """)

st.caption("Educational demonstration. Predictions are based on the classic Iris dataset.")
