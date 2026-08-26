"""
Stock Sentiment Analysis Dashboard

Run this application from PowerShell with:

    python -m streamlit run streamlit_app.py
"""

# Models:
# 1. SVM + TF-IDF
# 2. Bidirectional LSTM (BiLSTM)
# 3. BERT
#
# Dataset distribution:
# Positive = 40%, Negative = 30%, Neutral = 30%
#
# Evaluation:
# - Accuracy, Precision, Recall, F1-Score, Classification Report, Confusion Matrix, Training Time
#
# Early Stopping:
# - SVM: Not applicable
# - BiLSTM: Monitor validation loss
# - BERT: Monitor validation Macro F1

# ============================================================
# PART 1: IMPORT LIBRARIES
# ============================================================

# Streamlit is used to create the dashboard
import streamlit as st

# Pandas is used to load and manage dataset
import pandas as pd

# NumPy is used for numerical calculations
import numpy as np

# Matplotlib is used to display graphs
import matplotlib.pyplot as plt

# Used to calculate model training time
import time

# PyTorch is required for BERT
import torch

# ============================================================
# SCIKIT-LEARN LIBRARIES
# ============================================================

# Split dataset into training and testing data
from sklearn.model_selection import train_test_split

# Convert text into TF-IDF numerical features
from sklearn.feature_extraction.text import TfidfVectorizer

# Support Vector Machine
from sklearn.svm import LinearSVC

# Convert sentiment labels to numerical labels
from sklearn.preprocessing import LabelEncoder

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# TENSORFLOW / KERAS LIBRARIES FOR BiLSTM
# ============================================================

# Convert words into integer sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Make all sequences the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sequential neural network
from tensorflow.keras.models import Sequential

# Neural network layers
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Dropout
)

# Stop neural-network training when validation performance
# stops improving
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# HUGGING FACE / BERT LIBRARIES
# ============================================================

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# ============================================================
# PART 2: STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Sentiment Model Comparison",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Sentiment Analysis Dashboard")

st.write(
    """
    This dashboard compares three sentiment classification models:

    **SVM + TF-IDF vs BiLSTM vs BERT**

    The models are evaluated using the same balanced stock sentiment
    dataset and the same final test set.
    """
)

# ============================================================
# PART 3: SIDEBAR SETTINGS AND MODEL EXPLANATION
# ============================================================

with st.sidebar:

    st.header("⚙️ Training Settings")

    # --------------------------------------------------------
    # BiLSTM maximum epochs
    # --------------------------------------------------------

    bilstm_epochs = st.slider(
        "Maximum BiLSTM Epochs",
        min_value=3,
        max_value=20,
        value=10
    )

    # --------------------------------------------------------
    # BERT maximum epochs
    # --------------------------------------------------------

    bert_epochs = st.slider(
        "Maximum BERT Epochs",
        min_value=2,
        max_value=10,
        value=5
    )

    st.divider()

    st.header("📚 Model Explanation")

    # ========================================================
    # MODEL 1 EXPLANATION
    # ========================================================

    with st.expander("1️⃣ SVM + TF-IDF"):

        st.write(
            """
            ### SVM + TF-IDF

            **Type:** Traditional Machine Learning

            **Role:** Baseline model

            ### TF-IDF

            TF-IDF converts text into numerical features.

            It gives more importance to useful words and less
            importance to very common words.

            This project uses:

            - Unigram features
            - Bigram features
            - English stop-word removal

            ### SVM

            Support Vector Machine finds a decision boundary that
            separates:

            - Negative
            - Neutral
            - Positive

            ### Advantages

            - Fast training
            - Effective for text classification
            - Works well with high-dimensional data
            - Good baseline model

            ### Limitations

            - Does not deeply understand word order
            - Does not understand context like BERT

            ### Early Stopping

            Not applied because LinearSVC does not train using
            neural-network epochs.
            """
        )

    # ========================================================
    # MODEL 2 EXPLANATION
    # ========================================================

    with st.expander("2️⃣ BiLSTM"):

        st.write(
            """
            ### Bidirectional LSTM

            **Type:** Deep Learning

            BiLSTM learns patterns based on the sequence of words.

            ### Process

            1. Tokenizer converts words into numbers.
            2. Embedding converts numbers into dense vectors.
            3. Forward LSTM reads left → right.
            4. Backward LSTM reads right → left.
            5. Both directions are combined.
            6. Dense layers classify sentiment.

            ### Advantages

            - Learns word sequence
            - Learns contextual relationships
            - Can identify more complex text patterns

            ### Limitations

            - Slower than SVM
            - Requires more computation
            - Can overfit

            ### Early Stopping

            BiLSTM monitors **validation loss**.

            If validation loss does not improve for two epochs,
            training automatically stops.

            The best model weights are restored.
            """
        )

    # ========================================================
    # MODEL 3 EXPLANATION
    # ========================================================

    with st.expander("3️⃣ BERT"):

        st.write(
            """
            ### BERT

            **BERT = Bidirectional Encoder Representations
            from Transformers**

            **Type:** Transformer NLP model

            BERT is already pretrained on a large amount of text.

            We fine-tune it using the stock sentiment dataset.

            ### Process

            1. BERT tokenizer processes the sentence.
            2. Words are converted into tokens.
            3. Attention mechanisms examine word relationships.
            4. BERT considers surrounding context.
            5. Final classification layer predicts sentiment.

            ### Advantages

            - Strong contextual understanding
            - Pretrained language knowledge
            - Powerful for NLP classification

            ### Limitations

            - Slow on CPU
            - Requires more memory
            - GPU is recommended

            ### Early Stopping

            Validation **Macro F1-score** is checked after
            every epoch.

            Training stops when Macro F1 no longer improves.
            """
        )

# ============================================================
# PART 4: LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    # --------------------------------------------------------
    # Read balanced dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        "stock_sentiment_balanced_40_30_30_unique.csv"
    )

    # Remove missing sentences or labels
    df = df.dropna(
        subset=["Sentence", "Sentiment"]
    )

    # Convert sentences to string
    df["Sentence"] = (
        df["Sentence"]
        .astype(str)
        .str.strip()
    )

    # Standardise sentiment labels
    df["Sentiment"] = (
        df["Sentiment"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Keep only valid sentiment labels
    df = df[
        df["Sentiment"].isin(
            ["negative", "neutral", "positive"]
        )
    ]

    # --------------------------------------------------------
    # Remove duplicate sentences
    #
    # Your new dataset should already contain no duplicates,
    # but this is an extra safety check.
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["Sentence"]
    )

    # Reset row number
    df = df.reset_index(drop=True)

    return df

# Load the dataset
df = load_data()

# ============================================================
# PART 5: DATASET OVERVIEW
# ============================================================

st.header("1. Dataset Overview")


# Calculate total records for each class
class_counts = (
    df["Sentiment"]
    .value_counts()
    .reindex(
        ["positive", "negative", "neutral"]
    )
)

# Calculate percentages
class_percentage = (
    df["Sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .reindex(
        ["positive", "negative", "neutral"]
    )
)

# ============================================================
# DATASET METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{len(df):,}"
)

col2.metric(
    "Positive",
    f"{class_counts['positive']:,}"
)

col3.metric(
    "Negative",
    f"{class_counts['negative']:,}"
)

col4.metric(
    "Neutral",
    f"{class_counts['neutral']:,}"
)

# ============================================================
# DATASET PREVIEW
# ============================================================

left, right = st.columns([2, 1])

with left:

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

with right:

    st.subheader("Class Distribution")

    distribution_df = pd.DataFrame(
        {
            "Sentiment": class_counts.index,
            "Count": class_counts.values,
            "Percentage": class_percentage.values
        }
    )

    st.dataframe(
        distribution_df.round(2),
        hide_index=True,
        use_container_width=True
    )

    st.bar_chart(
        distribution_df.set_index(
            "Sentiment"
        )["Count"]
    )

# ============================================================
# PART 6: TRAIN / TEST SPLIT
# ============================================================

st.header("2. Train-Test Split")


# Input sentences
X = df["Sentence"]

# Target sentiment labels
y = df["Sentiment"]

# ------------------------------------------------------------
# 80% training data
# 20% testing data
#
# stratify=y maintains approximately the same sentiment
# distribution in the training and testing sets.
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

split_col1, split_col2 = st.columns(2)

split_col1.metric(
    "Training Records (80%)",
    f"{len(X_train):,}"
)

split_col2.metric(
    "Testing Records (20%)",
    f"{len(X_test):,}"
)

st.info(
    """
    All three models use the **same final testing dataset**.
    This provides a fairer comparison between SVM, BiLSTM and BERT.
    """
)

# ============================================================
# PART 7: COMMON MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    y_prediction,
    training_time
):

    """
    Calculate classification metrics.

    Macro averaging is used because each sentiment class
    receives equal importance.
    """

    accuracy = accuracy_score(
        y_true,
        y_prediction
    )

    precision = precision_score(
        y_true,
        y_prediction,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_prediction,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_prediction,
        average="macro",
        zero_division=0
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Training Time (s)": training_time
    }

# ============================================================
# PART 8: MODEL 1 - SVM + TF-IDF
# ============================================================

def train_svm():

    """
    Train SVM using TF-IDF text features.
    """
    start_time = time.time()

    # --------------------------------------------------------
    # TF-IDF VECTORIZER
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(

        # Use single words and two-word combinations
        ngram_range=(1, 2),

        # Ignore words appearing in more than 90% of documents
        max_df=0.90,

        # Ignore words appearing fewer than 3 times
        min_df=3,

        # Remove common English words
        stop_words="english",

        # Limit total number of TF-IDF features
        max_features=30000,

        lowercase=True
    )

    # Learn TF-IDF vocabulary from training data only
    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    # Apply same vocabulary to testing data
    X_test_tfidf = vectorizer.transform(
        X_test
    )

    # --------------------------------------------------------
    # CREATE SVM MODEL
    # --------------------------------------------------------

    svm_model = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    )

    # Train model
    svm_model.fit(
        X_train_tfidf,
        y_train
    )

    # Predict testing data
    predictions = svm_model.predict(
        X_test_tfidf
    )

    training_time = (
        time.time() - start_time
    )

    return (
        svm_model,
        vectorizer,
        predictions,
        training_time
    )

# ============================================================
# PART 9: MODEL 2 - BiLSTM
# ============================================================

def train_bilstm():

    """
    Train a Bidirectional LSTM neural network.
    """

    start_time = time.time()

    # ========================================================
    # CREATE INTERNAL TRAINING / VALIDATION SPLIT
    # ========================================================

    # We do NOT use the final test set for early stopping.
    #
    # 90% of original training data = model training
    # 10% of original training data = validation

    (
        X_lstm_train,
        X_lstm_val,
        y_lstm_train,
        y_lstm_val
    ) = train_test_split(

        X_train,
        y_train,

        test_size=0.10,

        random_state=42,

        stratify=y_train
    )

    # ========================================================
    # LABEL ENCODING
    # ========================================================

    encoder = LabelEncoder()

    # Learn the sentiment classes
    encoder.fit(y_train)

    y_lstm_train_encoded = encoder.transform(
        y_lstm_train
    )

    y_lstm_val_encoded = encoder.transform(
        y_lstm_val
    )

    # ========================================================
    # TOKENIZER SETTINGS
    # ========================================================

    max_words = 20000

    max_length = 100


    tokenizer = Tokenizer(
        num_words=max_words,
        oov_token="<OOV>"
    )

    # Learn vocabulary from training data only
    tokenizer.fit_on_texts(
        X_lstm_train
    )

    # Convert text to number sequences
    train_sequences = tokenizer.texts_to_sequences(
        X_lstm_train
    )

    validation_sequences = tokenizer.texts_to_sequences(
        X_lstm_val
    )

    test_sequences = tokenizer.texts_to_sequences(
        X_test
    )

    # ========================================================
    # PAD SEQUENCES
    # ========================================================

    X_lstm_train_pad = pad_sequences(
        train_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    X_lstm_val_pad = pad_sequences(
        validation_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    X_test_pad = pad_sequences(
        test_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    # ========================================================
    # BUILD BiLSTM MODEL
    # ========================================================

    bilstm_model = Sequential(

        [

            # Convert word IDs into dense vectors
            Embedding(
                input_dim=max_words,
                output_dim=128
            ),

            # Read sentence forward and backward
            Bidirectional(
                LSTM(
                    64,
                    return_sequences=False
                )
            ),

            # Reduce overfitting
            Dropout(0.5),

            # Hidden layer
            Dense(
                64,
                activation="relu"
            ),

            Dropout(0.5),

            # Three sentiment classes
            Dense(
                3,
                activation="softmax"
            )
        ]
    )

    # ========================================================
    # COMPILE MODEL
    # ========================================================

    bilstm_model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = EarlyStopping(

        # Monitor validation loss
        monitor="val_loss",

        # Stop after 2 epochs without improvement
        patience=2,

        # Minimum improvement required
        min_delta=0.001,

        # Restore weights from best epoch
        restore_best_weights=True,

        verbose=1
    )

    # ========================================================
    # TRAIN BiLSTM
    # ========================================================

    history = bilstm_model.fit(

        X_lstm_train_pad,
        y_lstm_train_encoded,

        validation_data=(
            X_lstm_val_pad,
            y_lstm_val_encoded
        ),

        # Maximum epochs
        epochs=bilstm_epochs,

        batch_size=64,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )

    # ========================================================
    # TEST PREDICTION
    # ========================================================

    prediction_probability = (
        bilstm_model.predict(
            X_test_pad,
            verbose=0
        )
    )

    prediction_numbers = np.argmax(
        prediction_probability,
        axis=1
    )

    predictions = encoder.inverse_transform(
        prediction_numbers
    )

    training_time = (
        time.time() - start_time
    )

    # Number of epochs actually completed
    epochs_completed = len(
        history.history["loss"]
    )

    return (
        bilstm_model,
        tokenizer,
        encoder,
        predictions,
        training_time,
        history,
        epochs_completed
    )

# ============================================================
# PART 10: CUSTOM DATASET FOR BERT
# ============================================================

class SentimentDataset(
    torch.utils.data.Dataset
):

    """
    Custom PyTorch Dataset used by Hugging Face Trainer.
    """

    def __init__(
        self,
        encodings,
        labels
    ):

        self.encodings = encodings

        self.labels = labels

    def __getitem__(
        self,
        index
    ):

        item = {

            key: torch.tensor(
                value[index]
            )

            for key, value
            in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item

    def __len__(self):

        return len(
            self.labels
        )

# ============================================================
# PART 11: BERT METRIC FUNCTION
# ============================================================

def compute_bert_metrics(
    eval_prediction
):

    """
    Calculate BERT validation metrics after each epoch.

    Macro F1 is used for early stopping.
    """

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {

        "accuracy": accuracy,

        "f1": macro_f1
    }

# ============================================================
# PART 12: MODEL 3 - BERT
# ============================================================

def train_bert():

    """
    Fine-tune BERT for three-class sentiment classification.
    """

    start_time = time.time()

    # ========================================================
    # INTERNAL TRAINING / VALIDATION SPLIT
    # ========================================================

    (
        X_bert_train,
        X_bert_val,
        y_bert_train,
        y_bert_val
    ) = train_test_split(

        X_train,
        y_train,

        test_size=0.10,

        random_state=42,

        stratify=y_train
    )

    # ========================================================
    # LABEL ENCODING
    # ========================================================

    encoder = LabelEncoder()

    encoder.fit(
        y_train
    )

    y_bert_train_encoded = encoder.transform(
        y_bert_train
    )

    y_bert_val_encoded = encoder.transform(
        y_bert_val
    )

    y_test_encoded = encoder.transform(
        y_test
    )

    # ========================================================
    # LOAD PRETRAINED BERT TOKENIZER
    # ========================================================

    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased"
    )

    # ========================================================
    # TOKENIZE TRAINING DATA
    # ========================================================

    train_encodings = tokenizer(

        X_bert_train.tolist(),

        truncation=True,

        padding=True,

        max_length=128
    )

    # ========================================================
    # TOKENIZE VALIDATION DATA
    # ========================================================

    validation_encodings = tokenizer(

        X_bert_val.tolist(),

        truncation=True,

        padding=True,

        max_length=128
    )

    # ========================================================
    # TOKENIZE TEST DATA
    # ========================================================

    test_encodings = tokenizer(

        X_test.tolist(),

        truncation=True,

        padding=True,

        max_length=128
    )

    # ========================================================
    # CREATE DATASETS
    # ========================================================

    train_dataset = SentimentDataset(

        train_encodings,

        y_bert_train_encoded.tolist()
    )

    validation_dataset = SentimentDataset(

        validation_encodings,

        y_bert_val_encoded.tolist()
    )

    test_dataset = SentimentDataset(

        test_encodings,

        y_test_encoded.tolist()
    )

    # ========================================================
    # LOAD PRETRAINED BERT MODEL
    # ========================================================

    bert_model = (
        BertForSequenceClassification
        .from_pretrained(

            "bert-base-uncased",

            num_labels=3
        )
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_arguments = TrainingArguments(

        # Folder to store checkpoints
        output_dir="./bert_results",

        # Maximum training epochs
        num_train_epochs=bert_epochs,

        # Training batch size
        per_device_train_batch_size=16,

        # Validation / test batch size
        per_device_eval_batch_size=32,

        # Standard BERT fine-tuning learning rate
        learning_rate=2e-5,

        # Regularisation
        weight_decay=0.01,

        # Evaluate after every epoch
        eval_strategy="epoch",

        # Save after every epoch
        save_strategy="epoch",

        # Restore best model automatically
        load_best_model_at_end=True,

        # Use Macro F1 as best-model metric
        metric_for_best_model="f1",

        # Higher F1 is better
        greater_is_better=True,

        # Keep only two checkpoints
        save_total_limit=2,

        # Reduce console output
        logging_steps=100,

        # Disable external logging
        report_to="none",

        # Use mixed precision when NVIDIA GPU is available
        fp16=torch.cuda.is_available(),

        # Reproducibility
        seed=42
    )

    # ========================================================
    # CREATE TRAINER
    # ========================================================

    trainer = Trainer(

        model=bert_model,

        args=training_arguments,

        train_dataset=train_dataset,

        # Validation set is used for early stopping
        eval_dataset=validation_dataset,

        # Calculate accuracy and Macro F1
        compute_metrics=compute_bert_metrics,

        # BERT early stopping
        callbacks=[

            EarlyStoppingCallback(

                # Stop after two evaluations
                # without improvement
                early_stopping_patience=2,

                # Minimum required F1 improvement
                early_stopping_threshold=0.001
            )
        ]
    )

    # ========================================================
    # FINE-TUNE BERT
    # ========================================================

    trainer.train()

    # ========================================================
    # TEST BERT
    # ========================================================

    prediction_output = trainer.predict(
        test_dataset
    )

    predicted_numbers = np.argmax(

        prediction_output.predictions,

        axis=1
    )

    predictions = encoder.inverse_transform(
        predicted_numbers
    )

    training_time = (
        time.time() - start_time
    )

    # Find epoch where training stopped
    completed_epoch = trainer.state.epoch

    # Find best validation Macro F1
    best_metric = trainer.state.best_metric

    return (

        bert_model,

        tokenizer,

        encoder,

        predictions,

        training_time,

        completed_epoch,

        best_metric
    )

# ============================================================
# PART 13: MODEL TRAINING SECTION
# ============================================================

st.header("3. Train and Compare Models")

# ------------------------------------------------------------
# Device information
# ------------------------------------------------------------

if torch.cuda.is_available():

    st.success(
        "🚀 NVIDIA GPU detected. BERT can use GPU acceleration."
    )

else:

    st.warning(
        """
        ⚠️ GPU was not detected.

        BERT will run on CPU and may train much more slowly.
        SVM and BiLSTM can still be trained normally.
        """
    )

# ============================================================
# TRAIN BUTTON
# ============================================================

if st.button(
    "🚀 Train & Compare All 3 Models",
    type="primary",
    use_container_width=True
):

    results = []

    predictions_dictionary = {}

    # ========================================================
    # MODEL 1: SVM
    # ========================================================

    with st.status(
        "Training SVM + TF-IDF...",
        expanded=True
    ) as svm_status:

        st.write(
            "Step 1: Creating TF-IDF features..."
        )

        st.write(
            "Step 2: Training Support Vector Machine..."
        )

        (
            svm_model,
            svm_vectorizer,
            svm_predictions,
            svm_time
        ) = train_svm()

        predictions_dictionary[
            "SVM + TF-IDF"
        ] = svm_predictions

        results.append(

            evaluate_model(

                "SVM + TF-IDF",

                y_test,

                svm_predictions,

                svm_time
            )
        )

        svm_status.update(
            label="✅ SVM + TF-IDF completed",
            state="complete"
        )

    # ========================================================
    # MODEL 2: BiLSTM
    # ========================================================

    with st.status(
        "Training BiLSTM...",
        expanded=True
    ) as lstm_status:

        st.write(
            "Step 1: Tokenizing sentences..."
        )

        st.write(
            "Step 2: Creating embedding sequences..."
        )

        st.write(
            "Step 3: Training Bidirectional LSTM..."
        )

        st.write(
            "Early stopping monitors validation loss."
        )

        (
            bilstm_model,
            bilstm_tokenizer,
            bilstm_encoder,
            bilstm_predictions,
            bilstm_time,
            bilstm_history,
            bilstm_epochs_completed
        ) = train_bilstm()

        predictions_dictionary[
            "BiLSTM"
        ] = bilstm_predictions

        results.append(

            evaluate_model(

                "BiLSTM",

                y_test,

                bilstm_predictions,

                bilstm_time
            )
        )

        lstm_status.update(
            label="✅ BiLSTM completed",
            state="complete"
        )

    # ========================================================
    # MODEL 3: BERT
    # ========================================================

    with st.status(
        "Training BERT...",
        expanded=True
    ) as bert_status:

        st.write(
            "Step 1: Loading pretrained BERT tokenizer..."
        )

        st.write(
            "Step 2: Tokenizing stock sentences..."
        )

        st.write(
            "Step 3: Fine-tuning BERT..."
        )

        st.write(
            "Early stopping monitors validation Macro F1."
        )

        (
            bert_model,
            bert_tokenizer,
            bert_encoder,
            bert_predictions,
            bert_time,
            bert_completed_epoch,
            bert_best_f1
        ) = train_bert()

        predictions_dictionary[
            "BERT"
        ] = bert_predictions

        results.append(

            evaluate_model(

                "BERT",

                y_test,

                bert_predictions,

                bert_time
            )
        )

        bert_status.update(
            label="✅ BERT completed",
            state="complete"
        )

    # ========================================================
    # PART 14: MODEL COMPARISON TABLE
    # ========================================================

    st.header(
        "4. Model Comparison Results"
    )


    results_df = pd.DataFrame(
        results
    )

    # Sort highest accuracy first
    results_df = results_df.sort_values(

        by="Accuracy",

        ascending=False

    ).reset_index(drop=True)

    # Add model ranking
    results_df.insert(

        0,

        "Rank",

        range(
            1,
            len(results_df) + 1
        )
    )

    # Create formatted display version
    display_results = results_df.copy()

    for metric in [

        "Accuracy",

        "Precision",

        "Recall",

        "F1-Score"

    ]:

        display_results[metric] = (

            display_results[metric]

            .map(
                lambda x: f"{x:.4f}"
            )
        )

    display_results[
        "Training Time (s)"
    ] = (

        display_results[
            "Training Time (s)"
        ]

        .map(
            lambda x: f"{x:.2f}"
        )
    )

    st.dataframe(

        display_results,

        hide_index=True,

        use_container_width=True
    )

    # ========================================================
    # PART 15: BEST MODEL
    # ========================================================

    best_model = results_df.iloc[0]

    st.success(
        f"""
        🏆 **Best Model: {best_model['Model']}**

        Accuracy: **{best_model['Accuracy']:.4f}**

        Precision: **{best_model['Precision']:.4f}**

        Recall: **{best_model['Recall']:.4f}**

        F1-Score: **{best_model['F1-Score']:.4f}**
        """
    )

    # ========================================================
    # PART 16: PERFORMANCE COMPARISON GRAPH
    # ========================================================

    st.subheader(
        "Performance Metric Comparison"
    )

    comparison_chart = (

        results_df

        .set_index("Model")

        [
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score"
            ]
        ]
    )

    st.bar_chart(
        comparison_chart
    )

    # ========================================================
    # PART 17: TRAINING TIME COMPARISON
    # ========================================================

    st.subheader(
        "Training Time Comparison"
    )

    time_chart = (

        results_df

        .set_index("Model")

        [["Training Time (s)"]]
    )

    st.bar_chart(
        time_chart
    )

    # ========================================================
    # PART 18: CONFUSION MATRIX
    # ========================================================

    st.header(
        "5. Confusion Matrix Comparison"
    )

    # Same order for all three models
    sentiment_labels = [

        "negative",

        "neutral",

        "positive"
    ]

    cm1, cm2, cm3 = st.columns(3)

    model_columns = [

        cm1,

        cm2,

        cm3
    ]

    model_names = [

        "SVM + TF-IDF",

        "BiLSTM",

        "BERT"
    ]

    for column, model_name in zip(

        model_columns,

        model_names
    ):

        with column:

            st.subheader(
                model_name
            )

            cm = confusion_matrix(

                y_test,

                predictions_dictionary[
                    model_name
                ],

                labels=sentiment_labels
            )

            fig, ax = plt.subplots(
                figsize=(5, 4)
            )

            display = ConfusionMatrixDisplay(

                confusion_matrix=cm,

                display_labels=sentiment_labels
            )

            display.plot(

                ax=ax,

                values_format="d",

                colorbar=False
            )

            ax.set_title(
                f"{model_name}\nConfusion Matrix"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

    # ========================================================
    # PART 19: CLASSIFICATION REPORTS
    # ========================================================

    st.header(
        "6. Detailed Classification Reports"
    )

    svm_tab, lstm_tab, bert_tab = st.tabs(

        [
            "SVM + TF-IDF",
            "BiLSTM",
            "BERT"
        ]
    )

    report_tabs = [

        svm_tab,

        lstm_tab,

        bert_tab
    ]

    for tab, model_name in zip(

        report_tabs,

        model_names
    ):

        with tab:

            report = classification_report(
                y_test,

                predictions_dictionary[
                    model_name
                ],

                labels=sentiment_labels,

                target_names=sentiment_labels,

                output_dict=True,

                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(

                report_df.round(4),

                use_container_width=True
            )

    # ========================================================
    # PART 20: EARLY STOPPING RESULTS
    # ========================================================

    st.header(
        "7. Early Stopping Results"
    )

    early1, early2, early3 = st.columns(3)

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    with early1:

        st.subheader(
            "SVM + TF-IDF"
        )

        st.info(
            """
            **Early Stopping: Not Applied**

            LinearSVC is not trained using epochs.

            Therefore validation-based early stopping is
            not required.
            """
        )

    # --------------------------------------------------------
    # BiLSTM
    # --------------------------------------------------------

    with early2:

        st.subheader(
            "BiLSTM"
        )

        st.write(
            f"""
            Maximum epochs: **{bilstm_epochs}**

            Completed epochs:
            **{bilstm_epochs_completed}**

            Monitor: **Validation Loss**

            Patience: **2**

            Minimum improvement: **0.001**
            """
        )

        if (
            bilstm_epochs_completed
            <
            bilstm_epochs
        ):
            st.success(
                "✅ Early stopping activated."
            )

        else:
            st.info(
                "BiLSTM reached the maximum epochs."
            )

    # --------------------------------------------------------
    # BERT
    # --------------------------------------------------------

    with early3:

        st.subheader(
            "BERT"
        )

        st.write(
            f"""
            Maximum epochs: **{bert_epochs}**

            Training stopped around epoch:
            **{bert_completed_epoch:.2f}**

            Monitor:
            **Validation Macro F1**

            Patience: **2**

            Minimum improvement: **0.001**
            """
        )

        if bert_best_f1 is not None:

            st.success(
                f"""
                Best validation Macro F1:

                **{bert_best_f1:.4f}**
                """
            )

    # ========================================================
    # PART 21: BiLSTM TRAINING HISTORY
    # ========================================================

    st.header(
        "8. BiLSTM Training History"
    )

    history_df = pd.DataFrame(
        bilstm_history.history
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    st.subheader(
        "BiLSTM Training vs Validation Accuracy"
    )

    accuracy_history = history_df[

        [
            "accuracy",
            "val_accuracy"
        ]
    ].copy()

    accuracy_history.columns = [

        "Training Accuracy",

        "Validation Accuracy"
    ]


    st.line_chart(
        accuracy_history
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    st.subheader(
        "BiLSTM Training vs Validation Loss"
    )

    loss_history = history_df[

        [
            "loss",
            "val_loss"
        ]
    ].copy()


    loss_history.columns = [

        "Training Loss",

        "Validation Loss"
    ]

    st.line_chart(
        loss_history
    )

    # ========================================================
    # PART 22: MODEL EXPLANATION TABLE
    # ========================================================

    st.header(
        "9. Model Explanation Summary"
    )

    explanation_df = pd.DataFrame(

        {

            "Model": [

                "SVM + TF-IDF",

                "BiLSTM",

                "BERT"
            ],

            "Model Type": [

                "Machine Learning",

                "Deep Learning",

                "Transformer NLP"
            ],

            "Text Representation": [

                "TF-IDF",

                "Embedding",

                "BERT Token Embeddings"
            ],

            "Context Understanding": [

                "Low",

                "Medium",

                "High"
            ],

            "Training Speed": [

                "Fast",

                "Medium",

                "Slow"
            ],

            "Early Stopping": [

                "Not Applicable",

                "Validation Loss",

                "Validation Macro F1"
            ]
        }
    )

    st.dataframe(

        explanation_df,

        hide_index=True,

        use_container_width=True
    )

    # ========================================================
    # PART 23: INTERPRETATION GUIDE
    # ========================================================

    st.header(
        "10. How to Interpret the Results"
    )

    st.write(
        """
        ### Accuracy

        Accuracy measures the percentage of all test samples
        classified correctly.

        Higher accuracy indicates better overall classification.

        ---

        ### Precision

        Precision measures how many predictions made for a
        sentiment class were actually correct.

        High precision means fewer false-positive predictions.

        ---

        ### Recall

        Recall measures how many actual samples of a sentiment
        class were successfully detected.

        High recall means the model misses fewer samples.

        ---

        ### F1-Score

        F1-score combines precision and recall.

        Macro F1 gives equal importance to negative, neutral,
        and positive sentiment.

        ---

        ### Confusion Matrix

        The diagonal values show correct predictions.

        Values outside the diagonal represent
        misclassification.

        ---

        ### Training Time

        Training time represents computational efficiency.

        A model with slightly lower accuracy but much faster
        training can still be useful depending on the
        application.
        """
    )

    # ========================================================
    # PART 24: FINAL MODEL INTERPRETATION
    # ========================================================

    st.header(
        "11. Final Model Interpretation"
    )

    st.write(
        """
        ### SVM + TF-IDF

        SVM + TF-IDF is used as the baseline model.

        It converts stock-related sentences into numerical
        TF-IDF features and finds decision boundaries between
        the three sentiment classes.

        It is expected to train significantly faster than
        deep-learning models.

        ### BiLSTM

        BiLSTM considers the sequential order of words.

        The bidirectional architecture allows the model to
        process information from both directions of a sentence.

        Early stopping reduces the risk of overfitting.

        ### BERT

        BERT is a pretrained Transformer language model.

        It uses contextual word representations and
        self-attention, allowing it to interpret words based on
        the surrounding sentence.

        BERT is usually more computationally expensive than SVM
        and BiLSTM.

        ### Final Selection

        The best model should not be selected using accuracy
        alone.

        The comparison should consider:

        - Accuracy
        - Precision
        - Recall
        - Macro F1-score
        - Confusion matrix
        - Training time
        - Model complexity
        """
    )