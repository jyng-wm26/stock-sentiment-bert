"""Streamlit dashboard for three-class BERT sentiment analysis."""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# Keep the same label mapping used in the original project.
LABEL2ID = {"negative": 0, "positive": 1, "neutral": 2}
ID2LABEL = {0: "negative", 1: "positive", 2: "neutral"}
LABEL_NAMES = ["negative", "positive", "neutral"]
MODEL_NAME = "google-bert/bert-base-uncased"


def clean_data(uploaded_file) -> pd.DataFrame:
    """Read, validate, clean, and encode the uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    required = {"Sentence", "Sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df[["Sentence", "Sentiment"]].dropna().drop_duplicates().copy()
    df["Sentence"] = df["Sentence"].astype(str).str.strip()
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()
    df = df[df["Sentence"].ne("")]

    unknown = sorted(set(df["Sentiment"]) - set(LABEL2ID))
    if unknown:
        raise ValueError(f"Unsupported labels found: {', '.join(unknown)}")

    df["label"] = df["Sentiment"].map(LABEL2ID).astype(int)
    return df.reset_index(drop=True)


def calculate_metrics(eval_prediction):
    """Calculate accuracy and weighted/macro precision, recall, and F1."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f_w,
        "precision_macro": p_m,
        "recall_macro": r_m,
        "f1_macro": f_m,
    }


def train_bert(df: pd.DataFrame, epochs: int, batch_size: int):
    """Split the data, fine-tune BERT, and evaluate it on unseen test data."""
    seed = 42
    set_seed(seed)

    # Create stratified 70% training, 15% validation, and 15% testing sets.
    train_df, temporary_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=seed,
        stratify=temporary_df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def make_dataset(frame):
        dataset = Dataset.from_pandas(
            frame[["Sentence", "label"]].reset_index(drop=True),
            preserve_index=False,
        )
        return dataset.map(
            lambda batch: tokenizer(
                batch["Sentence"], truncation=True, max_length=128
            ),
            batched=True,
            remove_columns=["Sentence"],
        )

    train_dataset = make_dataset(train_df)
    validation_dataset = make_dataset(validation_df)
    test_dataset = make_dataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    # Use a temporary checkpoint folder because Streamlit sessions are temporary.
    output_directory = tempfile.mkdtemp(prefix="bert_streamlit_")
    training_arguments = TrainingArguments(
        output_dir=output_directory,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        # Use warmup_steps for compatibility across Transformers versions.
        warmup_steps=0,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=calculate_metrics,
    )
    trainer.train()

    test_output = trainer.predict(test_dataset)
    actual = test_output.label_ids
    predicted = np.argmax(test_output.predictions, axis=-1)
    metrics = calculate_metrics((test_output.predictions, actual))
    report = classification_report(
        actual,
        predicted,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2])

    return {
        "trainer": trainer,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "report": report,
        "matrix": matrix,
        "sizes": (len(train_df), len(validation_df), len(test_df)),
    }


def predict_sentence(sentence: str, trainer, tokenizer):
    """Predict the sentiment and probabilities of one new sentence."""
    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, max_length=128
    )
    device = trainer.model.device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    prediction_id = int(np.argmax(probabilities))
    return ID2LABEL[prediction_id], probabilities


# Configure the Streamlit page.
st.set_page_config(page_title="BERT Sentiment Analysis", page_icon="📊", layout="wide")
st.title("BERT Financial Sentiment Analysis")
st.write("Classify sentences as **negative**, **positive**, or **neutral**.")

# Explain the numeric labels used by the model.
st.info("Label mapping: negative = 0, positive = 1, neutral = 2")

# Put training controls in the sidebar.
with st.sidebar:
    st.header("Training settings")
    epochs = st.slider("Epochs", min_value=1, max_value=4, value=2)
    batch_size = st.selectbox("Batch size", options=[4, 8, 16], index=1)
    st.caption("Smaller values use less memory but take longer.")

# Let the user upload the project dataset.
uploaded_file = st.file_uploader(
    "Upload data.csv",
    type=["csv"],
    help="The file must contain Sentence and Sentiment columns.",
)

if uploaded_file is not None:
    try:
        dataframe = clean_data(uploaded_file)
    except Exception as error:
        st.error(str(error))
        st.stop()

    st.subheader("Dataset overview")
    column1, column2 = st.columns([2, 1])
    with column1:
        st.dataframe(dataframe[["Sentence", "Sentiment"]].head(20), use_container_width=True)
    with column2:
        st.write("Class distribution")
        st.bar_chart(dataframe["Sentiment"].value_counts())
        st.metric("Clean records", f"{len(dataframe):,}")

    if st.button("Train BERT model", type="primary"):
        with st.spinner("Fine-tuning BERT. This may take several minutes..."):
            try:
                st.session_state.bert_results = train_bert(
                    dataframe, epochs=epochs, batch_size=batch_size
                )
            except Exception as error:
                st.exception(error)
                st.stop()
        st.success("BERT training and testing completed.")

# Show results after training and preserve them during Streamlit reruns.
if "bert_results" in st.session_state:
    results = st.session_state.bert_results
    metrics = results["metrics"]
    train_size, validation_size, test_size = results["sizes"]

    st.subheader("Model evaluation results")
    st.caption(
        f"Training: {train_size:,} | Validation: {validation_size:,} | Test: {test_size:,}"
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
    metric_columns[1].metric("Precision", f"{metrics['precision_weighted']:.4f}")
    metric_columns[2].metric("Recall", f"{metrics['recall_weighted']:.4f}")
    metric_columns[3].metric("F1-score", f"{metrics['f1_weighted']:.4f}")

    tab1, tab2, tab3 = st.tabs(
        ["Classification report", "Confusion matrix", "Test a sentence"]
    )

    with tab1:
        report_dataframe = pd.DataFrame(results["report"]).transpose()
        st.dataframe(report_dataframe.round(4), use_container_width=True)

    with tab2:
        figure, axis = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            results["matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            ax=axis,
        )
        axis.set_title("BERT Sentiment Confusion Matrix")
        axis.set_xlabel("Predicted sentiment")
        axis.set_ylabel("Actual sentiment")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    with tab3:
        sentence = st.text_area(
            "Enter a financial sentence",
            placeholder="Example: The company reported higher profits this quarter.",
        )
        if st.button("Predict sentiment"):
            if not sentence.strip():
                st.warning("Please enter a sentence first.")
            else:
                label, probabilities = predict_sentence(
                    sentence,
                    results["trainer"],
                    results["tokenizer"],
                )
                st.success(f"Predicted sentiment: {label.upper()}")
                probability_table = pd.DataFrame(
                    {"Sentiment": LABEL_NAMES, "Probability": probabilities}
                )
                st.dataframe(probability_table, use_container_width=True)
