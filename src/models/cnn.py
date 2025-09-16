# Import Modules
from typing import List
import os, json, joblib, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from src.logger import logger

class CNNIDS: # Define CNN Class
    def __init__(self, max_len: int = 100, vocab_size: int = 20000, oov_token: str = "<OOV>", dropout_rate: float = 0.3):
        self.max_len = max_len # Ficed sequence padding length
        self.vocab_size = vocab_size # Max vocab size
        self.oov_token = oov_token # Token used for unseen keywords
        self.dropout_rate = dropout_rate # Define dropout rate
        self.tokenizer: Tokenizer = None # Tokenizer placeholder
        self.model: tf.keras.Model = None # Model placeholder

    def _build(self): # Create CNN
        model = models.Sequential([
            layers.Input(shape=(self.max_len,)), # Fixed length int token
            layers.Embedding(self.vocab_size + 1, 128), # Word embeds
            layers.Conv1D(128, 5, padding="same", activation="relu"), # Detect ngrams
            layers.GlobalMaxPooling1D(), # Maintain strongest activiation for filter
            layers.Dense(64, activation="relu"), # Combine Features
            layers.Dropout(self.dropout_rate), # Regulate overfitting
            layers.Dense(1, activation="sigmoid"), # Malicious outcome
        ])
        model.compile( # Create report
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]
        )
        self.model = model

    def _texts_to_sequences(self, texts: List[str]): # Text preprocessong
        seqs = self.tokenizer.texts_to_sequences(texts) # Tokenize strings to ints
        return pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post") # Padding

    def fit(self, train_texts: List[str], train_y, val_texts: List[str], val_y, epochs=10, batch_size=64): # Model training teokenizer
        logger.info("[*] Fitting tokenizer")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(train_texts)

        X_tr = self._texts_to_sequences(train_texts) # Fit cmd texts
        X_va = self._texts_to_sequences(val_texts) # Fit validation texts

        logger.info("[*] Building CNN") # Log to serial
        self._build()

        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)] # Intelligence fallback
        logger.info("[*] Training CNN")
        self.model.fit(X_tr, train_y, validation_data=(X_va, val_y), epochs=epochs, batch_size=batch_size, verbose=2)
                       #callbacks=cb)

    def evaluate(self, test_texts: List[str], test_y): # Evaluate unssen
        X_te = self._texts_to_sequences(test_texts) # Covert test texts
        preds = (self.model.predict(X_te, verbose=0).ravel() >= 0.5).astype(int)
        print(classification_report(test_y, preds, digits=4)) # Produce report

    def predict(self, texts: List[str]): # Geneate probabilities
        X = self._texts_to_sequences(texts) # Convert test texts
        probs = self.model.predict(X, verbose=0).ravel()
        return (probs >= 0.5).astype(int) # Produce probability

    def save(self, model_path: str, tokenizer_path: str, meta_path: str): # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        joblib.dump(self.tokenizer, tokenizer_path) # Creatre joblib file
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"max_len": self.max_len, "vocab_size": self.vocab_size, "oov_token": self.oov_token, "dropout_rate": self.dropout_rate}, f)

    def load(self, model_path: str, tokenizer_path: str, meta_path: str): # Load model
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.max_len = meta.get("max_len", self.max_len)
        self.vocab_size = meta.get("vocab_size", self.vocab_size)
        self.oov_token = meta.get("oov_token", self.oov_token)
        self.dropout_rate = meta.get("dropout_rate", self.dropout_rate)
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
