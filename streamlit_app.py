"""Streamlit dashboard for balanced three-class BERT stock sentiment analysis.

Run this application from PowerShell with:

    python -m streamlit run streamlit_app.py

The program performs these main tasks:
1. Reads an uploaded CSV using several possible text encodings.
2. Cleans missing, invalid, duplicate, and conflicting records.
3. Splits the clean data into training, validation, and testing sets.
4. Balances only the training set using hybrid under/oversampling.
5. Fine-tunes BERT and displays evaluation results.
6. Predicts the sentiment of a new financial sentence.
"""


# =============================================================================
# PART 1: IMPORT THE REQUIRED LIBRARIES
# =============================================================================

import hashlib
import inspect
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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


# =============================================================================
# PART 2: DEFINE THE MODEL AND SENTIMENT LABELS
# =============================================================================

# Keep one fixed mapping throughout cleaning, training, evaluation, and prediction.
LABEL2ID = {"negative": 0, "positive": 1, "neutral": 2}
ID2LABEL = {0: "negative", 1: "positive", 2: "neutral"}
LABEL_NAMES = ["negative", "positive", "neutral"]

# This is the standard uncased English BERT model from Hugging Face.
MODEL_NAME = "google-bert/bert-base-uncased"

# A fixed seed makes splitting, balancing, and training more reproducible.
RANDOM_SEED = 42


# =============================================================================
# PART 3: READ THE CSV WITH ENCODING FALLBACK
# =============================================================================

def read_csv_with_fallback(uploaded_file):
    """Read a CSV by trying common encodings used by the project dataset."""

    # datav1.csv uses CP1252, while other uploaded files may use UTF-8.
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None

    for encoding in encodings:
        try:
            # Reset the uploaded file before every reading attempt.
            uploaded_file.seek(0)
            dataframe = pd.read_csv(uploaded_file, encoding=encoding)
            return dataframe, encoding
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        "The CSV could not be read using UTF-8, CP1252, or Latin-1 encoding."
    ) from last_error


# =============================================================================
# PART 4: CLEAN AND VALIDATE THE DATASET
# =============================================================================

def clean_data(uploaded_file):
    """Clean the stock sentiment dataset and return cleaning information."""

    df, detected_encoding = read_csv_with_fallback(uploaded_file)
    raw_row_count = len(df)

    # The application needs these exact two source columns.
    required_columns = {"Sentence", "Sentiment"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Keep only the columns needed by the model.
    df = df[["Sentence", "Sentiment"]].copy()

    # Count and remove records with a missing sentence or sentiment label.
    missing_mask = df["Sentence"].isna() | df["Sentiment"].isna()
    missing_removed = int(missing_mask.sum())
    df = df[~missing_mask].copy()

    # Remove extra spaces and standardise label capitalisation.
    df["Sentence"] = df["Sentence"].astype(str).str.strip()
    df["Sentiment"] = (
        df["Sentiment"].astype(str).str.strip().str.lower()
    )

    # Remove empty sentences after whitespace has been stripped.
    empty_sentence_mask = df["Sentence"].eq("")
    empty_sentences_removed = int(empty_sentence_mask.sum())
    df = df[~empty_sentence_mask].copy()

    # Remove malformed labels and keep only the intended three classes.
    valid_label_mask = df["Sentiment"].isin(LABEL2ID)
    invalid_labels_removed = int((~valid_label_mask).sum())
    df = df[valid_label_mask].copy()

    # Create a temporary key so sentences differing only in case or spaces match.
    df["_sentence_key"] = (
        df["Sentence"]
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Identify sentences that have been assigned more than one sentiment label.
    labels_per_sentence = df.groupby("_sentence_key")["Sentiment"].nunique()
    conflicting_keys = labels_per_sentence[labels_per_sentence > 1].index
    conflicting_sentence_count = len(conflicting_keys)
    conflicting_row_mask = df["_sentence_key"].isin(conflicting_keys)
    conflicting_rows_removed = int(conflicting_row_mask.sum())

    # Remove all conflicting versions because their correct label is uncertain.
    df = df[~conflicting_row_mask].copy()

    # Keep one copy when the same sentence and label appear repeatedly.
    rows_before_deduplication = len(df)
    df = df.drop_duplicates(subset="_sentence_key", keep="first")
    duplicate_rows_removed = rows_before_deduplication - len(df)

    # The temporary comparison key is not needed by BERT.
    df = df.drop(columns="_sentence_key")

    # Convert text sentiment labels into the numeric IDs required by BERT.
    df["label"] = df["Sentiment"].map(LABEL2ID).astype(int)
    df = df.reset_index(drop=True)

    # Confirm that cleaning has not removed an entire sentiment class.
    missing_classes = set(LABEL2ID) - set(df["Sentiment"].unique())
    if missing_classes:
        raise ValueError(
            "The cleaned dataset is missing these classes: "
            + ", ".join(sorted(missing_classes))
        )

    # Return statistics so users can see exactly what the cleaning process did.
    cleaning_summary = {
        "encoding": detected_encoding,
        "raw_rows": raw_row_count,
        "clean_rows": len(df),
        "missing_removed": missing_removed,
        "empty_sentences_removed": empty_sentences_removed,
        "invalid_labels_removed": invalid_labels_removed,
        "conflicting_sentences": conflicting_sentence_count,
        "conflicting_rows_removed": conflicting_rows_removed,
        "duplicates_removed": duplicate_rows_removed,
    }

    return df, cleaning_summary


# =============================================================================
# PART 5: BALANCE ONLY THE TRAINING DATA
# =============================================================================

def balance_training_data(train_df, seed=RANDOM_SEED):
    """Balance training classes using small-scale under/oversampling.

    The median class size is used as the target:
    - Classes larger than the target are randomly undersampled.
    - Classes smaller than the target are randomly oversampled.

    Validation and testing data are never passed to this function.
    """

    class_counts = train_df["label"].value_counts()
    target_size = int(class_counts.median())
    balanced_groups = []

    for label_id, group in train_df.groupby("label"):
        balanced_group = group.sample(
            n=target_size,
            # Duplicate examples only when a class is smaller than the target.
            replace=len(group) < target_size,
            random_state=seed,
        )
        balanced_groups.append(balanced_group)

    # Join the three classes and shuffle their row order.
    balanced_df = pd.concat(balanced_groups, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed)

    return balanced_df.reset_index(drop=True)


# =============================================================================
# PART 6: CALCULATE MODEL EVALUATION METRICS
# =============================================================================

def calculate_metrics(eval_prediction):
    """Calculate accuracy plus weighted and macro evaluation scores."""

    logits, actual_labels = eval_prediction
    predicted_labels = np.argmax(logits, axis=-1)

    # Weighted scores consider the number of records in each class.
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            actual_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    # Macro scores give equal importance to all three sentiment classes.
    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            actual_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy_score(actual_labels, predicted_labels),
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


# =============================================================================
# PART 7: CONVERT A PANDAS DATAFRAME INTO A BERT DATASET
# =============================================================================

def make_bert_dataset(frame, tokenizer):
    """Tokenize sentences and create a Hugging Face Dataset."""

    dataset = Dataset.from_pandas(
        frame[["Sentence", "label"]].reset_index(drop=True),
        preserve_index=False,
    )

    def tokenize_batch(batch):
        # Truncation limits every sentence to at most 128 BERT tokens.
        return tokenizer(
            batch["Sentence"],
            truncation=True,
            max_length=128,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["Sentence"],
    )


# =============================================================================
# PART 8: BUILD VERSION-COMPATIBLE TRAINING ARGUMENTS
# =============================================================================

def create_training_arguments(output_directory, epochs, batch_size):
    """Create TrainingArguments compatible with old and new Transformers."""

    argument_values = {
        "output_dir": output_directory,
        "save_strategy": "no",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "seed": RANDOM_SEED,
        "logging_strategy": "epoch",
    }

    # New Transformers versions use eval_strategy; older ones use
    # evaluation_strategy. Checking the signature prevents a version error.
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        argument_values["eval_strategy"] = "epoch"
    else:
        argument_values["evaluation_strategy"] = "epoch"

    return TrainingArguments(**argument_values)


# =============================================================================
# PART 9: FINE-TUNE BERT AND TEST THE MODEL
# =============================================================================

def train_bert(df, epochs, batch_size):
    """Split, balance, fine-tune, and evaluate the BERT model."""

    set_seed(RANDOM_SEED)

    # First split: 70% training and 30% temporary data.
    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    # Second split: divide the temporary data into 15% validation and 15% test.
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_df["label"],
    )

    # Save the natural training counts before balancing for comparison.
    train_counts_before = (
        train_df["Sentiment"]
        .value_counts()
        .reindex(LABEL_NAMES, fill_value=0)
        .to_dict()
    )

    # Balance training data only; validation and test distributions stay natural.
    train_df = balance_training_data(train_df)

    train_counts_after = (
        train_df["Sentiment"]
        .value_counts()
        .reindex(LABEL_NAMES, fill_value=0)
        .to_dict()
    )

    # Download the tokenizer and convert text into BERT token IDs.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = make_bert_dataset(train_df, tokenizer)
    validation_dataset = make_bert_dataset(validation_df, tokenizer)
    test_dataset = make_bert_dataset(test_df, tokenizer)

    # Load BERT with a new classification layer containing three outputs.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    # Store temporary checkpoints outside the project folder.
    output_directory = tempfile.mkdtemp(prefix="bert_streamlit_")
    training_arguments = create_training_arguments(
        output_directory,
        epochs,
        batch_size,
    )

    # Configure the Hugging Face Trainer.
    trainer_options = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": calculate_metrics,
    }

    # New Transformers uses processing_class; older versions use tokenizer.
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_options["processing_class"] = tokenizer
    else:
        trainer_options["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_options)

    # Fine-tune BERT on the balanced training set.
    trainer.train()

    # Evaluate once on the untouched test set.
    test_output = trainer.predict(test_dataset)
    actual_labels = test_output.label_ids
    predicted_labels = np.argmax(test_output.predictions, axis=-1)

    metrics = calculate_metrics((test_output.predictions, actual_labels))

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2],
    )

    # Store only the objects required by the dashboard and prediction function.
    return {
        "model": trainer.model,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "report": report,
        "matrix": matrix,
        "sizes": {
            "training_balanced": len(train_df),
            "validation": len(validation_df),
            "testing": len(test_df),
        },
        "train_counts_before": train_counts_before,
        "train_counts_after": train_counts_after,
    }


# =============================================================================
# PART 10: PREDICT THE SENTIMENT OF ONE NEW SENTENCE
# =============================================================================

def predict_sentence(sentence, model, tokenizer):
    """Return the predicted sentiment and probability for every class."""

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    # Put the new input on the same CPU or GPU device as the model.
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    prediction_id = int(np.argmax(probabilities))
    return ID2LABEL[prediction_id], probabilities


# =============================================================================
# PART 11: CONFIGURE THE STREAMLIT PAGE
# =============================================================================

st.set_page_config(
    page_title="Balanced BERT Stock Sentiment Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("Balanced BERT Stock Sentiment Analysis")
st.write(
    "Upload a stock sentiment CSV, clean and balance its training data, "
    "fine-tune BERT, and evaluate positive, negative, and neutral predictions."
)

st.info("Label mapping: negative = 0, positive = 1, neutral = 2")


# =============================================================================
# PART 12: CREATE THE SIDEBAR TRAINING CONTROLS
# =============================================================================

with st.sidebar:
    st.header("Training settings")

    epochs = st.slider(
        "Epochs",
        min_value=1,
        max_value=4,
        value=2,
        help="More epochs may improve learning but increase training time.",
    )

    batch_size = st.selectbox(
        "Batch size",
        options=[4, 8, 16],
        index=1,
        help="Use 4 if your computer runs out of memory.",
    )

    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    st.write(f"Training device: **{device_name}**")
    st.caption("CPU training can take a long time. Do not close the terminal.")


# =============================================================================
# PART 13: UPLOAD, CLEAN, AND DISPLAY THE DATASET
# =============================================================================

uploaded_file = st.file_uploader(
    "Upload datav1.csv",
    type=["csv"],
    help="The CSV must contain Sentence and Sentiment columns.",
)

if uploaded_file is not None:
    try:
        dataframe, cleaning_summary = clean_data(uploaded_file)
    except Exception as error:
        st.error(f"Dataset error: {error}")
        st.stop()

    # Create a signature so results from a different uploaded dataset are cleared.
    dataset_bytes = dataframe[["Sentence", "Sentiment"]].to_csv(
        index=False
    ).encode("utf-8")
    dataset_signature = hashlib.sha256(dataset_bytes).hexdigest()

    if st.session_state.get("dataset_signature") != dataset_signature:
        st.session_state.pop("bert_results", None)
        st.session_state["dataset_signature"] = dataset_signature

    st.subheader("1. Cleaned dataset overview")

    overview_column, distribution_column = st.columns([2, 1])

    with overview_column:
        st.dataframe(
            dataframe[["Sentence", "Sentiment"]].head(20),
            use_container_width=True,
        )

    with distribution_column:
        class_distribution = (
            dataframe["Sentiment"]
            .value_counts()
            .reindex(LABEL_NAMES, fill_value=0)
        )
        st.write("Clean class distribution")
        st.bar_chart(class_distribution)
        st.metric("Clean records", f"{len(dataframe):,}")

    # Explain how many records were removed for each data-quality issue.
    with st.expander("View cleaning details"):
        st.write(f"Detected encoding: **{cleaning_summary['encoding']}**")
        st.write(f"Original records: **{cleaning_summary['raw_rows']:,}**")
        st.write(f"Clean records: **{cleaning_summary['clean_rows']:,}**")
        st.write(
            "Missing-label or missing-sentence records removed: "
            f"**{cleaning_summary['missing_removed']:,}**"
        )
        st.write(
            "Empty sentences removed: "
            f"**{cleaning_summary['empty_sentences_removed']:,}**"
        )
        st.write(
            "Invalid-label records removed: "
            f"**{cleaning_summary['invalid_labels_removed']:,}**"
        )
        st.write(
            "Conflicting sentences found: "
            f"**{cleaning_summary['conflicting_sentences']:,}**"
        )
        st.write(
            "Rows removed because of conflicting labels: "
            f"**{cleaning_summary['conflicting_rows_removed']:,}**"
        )
        st.write(
            "Same-label duplicate rows removed: "
            f"**{cleaning_summary['duplicates_removed']:,}**"
        )

    st.subheader("2. Train the BERT model")
    st.warning(
        "Training downloads BERT the first time. CPU training may take a long "
        "time; using a CUDA-compatible GPU is much faster."
    )

    if st.button("Train balanced BERT model", type="primary"):
        with st.spinner("Cleaning, balancing, and fine-tuning BERT..."):
            try:
                st.session_state["bert_results"] = train_bert(
                    dataframe,
                    epochs=epochs,
                    batch_size=batch_size,
                )
            except Exception as error:
                st.exception(error)
                st.stop()

        st.success("BERT training and testing completed.")


# =============================================================================
# PART 14: DISPLAY MODEL RESULTS
# =============================================================================

if "bert_results" in st.session_state:
    results = st.session_state["bert_results"]
    metrics = results["metrics"]
    sizes = results["sizes"]

    st.subheader("3. Model evaluation results")
    st.caption(
        f"Balanced training: {sizes['training_balanced']:,} | "
        f"Validation: {sizes['validation']:,} | "
        f"Unchanged test set: {sizes['testing']:,}"
    )

    # Display the most important overall evaluation measurements.
    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
    metric_columns[1].metric(
        "Macro precision",
        f"{metrics['precision_macro']:.4f}",
    )
    metric_columns[2].metric(
        "Macro recall",
        f"{metrics['recall_macro']:.4f}",
    )
    metric_columns[3].metric("Macro F1-score", f"{metrics['f1_macro']:.4f}")

    # Separate detailed results into easy-to-read tabs.
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Class balancing",
            "Classification report",
            "Confusion matrix",
            "Test a sentence",
        ]
    )

    with tab1:
        st.write("Training distribution before and after balancing")

        balancing_table = pd.DataFrame(
            {
                "Before balancing": results["train_counts_before"],
                "After balancing": results["train_counts_after"],
            }
        ).reindex(LABEL_NAMES)

        st.dataframe(balancing_table, use_container_width=True)
        st.bar_chart(balancing_table)
        st.caption(
            "Only the training set is balanced. Validation and test sets "
            "remain unchanged for realistic evaluation."
        )

    with tab2:
        report_dataframe = pd.DataFrame(results["report"]).transpose()
        st.dataframe(report_dataframe.round(4), use_container_width=True)

        st.write("Weighted measurements")
        st.write(
            {
                "Precision weighted": round(
                    metrics["precision_weighted"], 4
                ),
                "Recall weighted": round(metrics["recall_weighted"], 4),
                "F1 weighted": round(metrics["f1_weighted"], 4),
            }
        )

    with tab3:
        # Draw the confusion matrix without requiring the Seaborn package.
        figure, axis = plt.subplots(figsize=(7, 5))
        matrix_display = ConfusionMatrixDisplay(
            confusion_matrix=results["matrix"],
            display_labels=LABEL_NAMES,
        )
        matrix_display.plot(
            ax=axis,
            cmap="Blues",
            values_format="d",
            colorbar=False,
        )
        axis.set_title("BERT Stock Sentiment Confusion Matrix")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    with tab4:
        sentence = st.text_area(
            "Enter a financial sentence",
            placeholder=(
                "Example: The company reported higher profits this quarter."
            ),
        )

        if st.button("Predict sentiment"):
            if not sentence.strip():
                st.warning("Please enter a sentence first.")
            else:
                predicted_sentiment, probabilities = predict_sentence(
                    sentence,
                    results["model"],
                    results["tokenizer"],
                )

                st.success(
                    f"Predicted sentiment: {predicted_sentiment.upper()}"
                )

                probability_table = pd.DataFrame(
                    {
                        "Sentiment": LABEL_NAMES,
                        "Probability": probabilities,
                    }
                )
                probability_table["Probability"] = probability_table[
                    "Probability"
                ].map(lambda value: f"{value:.2%}")

                st.dataframe(
                    probability_table,
                    use_container_width=True,
                    hide_index=True,
                )


# =============================================================================
# PART 15: SHOW INSTRUCTIONS BEFORE A FILE IS UPLOADED
# =============================================================================

if uploaded_file is None and "bert_results" not in st.session_state:
    st.caption(
        "Begin by uploading a CSV containing the Sentence and Sentiment columns."
    )