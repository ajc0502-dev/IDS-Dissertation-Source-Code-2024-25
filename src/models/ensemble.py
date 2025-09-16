# Import modules
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os, joblib
from src.logger import logger

# Define Cowrie categories
TEXT_COL = "commands" # Text
NUMERIC_COLS = ["ip_encoded","num_commands","suspicious_token_hits","avg_command_len","duration"] # Numeric

# Create Ensemble
def build_ensemble(random_state: int = 42, # Initial state
                   tfidf_max_features: int = 20000, # Initial max features
                   tfidf_ngram_max: int = 2, # Initial ngram max
                   tfidf_min_df: int = 2, # Initial document frequency
                   rf_estimators: int = 300) -> Pipeline: # Initial RF trees
    text_vec = ("tfidf", TfidfVectorizer(
        ngram_range=(1, tfidf_ngram_max),
        min_df=tfidf_min_df,
        max_features=tfidf_max_features
    ), TEXT_COL)
    passthrough_nums = ("num", "passthrough", NUMERIC_COLS)

    preproc = ColumnTransformer(
        transformers=[text_vec, passthrough_nums],
        remainder="drop",
        verbose_feature_names_out=False
    )

    lr_text = ("lr_text", LogisticRegression(max_iter=8000, C=0.25, random_state=random_state)) # Define LR
    final_est = LogisticRegression(max_iter=8000, solver="saga", C=0.25, class_weight="balanced", random_state=random_state) # Define LR init
    rf_tab = ("rf_tab", RandomForestClassifier(n_estimators=rf_estimators, random_state=random_state, n_jobs=-1)) # Define random frorests
    gbdt_tab = ("gbdt_tab", GradientBoostingClassifier(random_state=random_state)) # Define GBC

    stack = StackingClassifier(
        estimators=[lr_text, rf_tab, gbdt_tab],
        final_estimator=final_est,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )

    model = Pipeline(steps=[("preproc", preproc), ("clf", stack)])
    return model

class EnsembleIDS: # Define ensemble class for imports with default values
    def __init__(self, random_state: int = 42,
                 tfidf_max_features: int = 10000,
                 tfidf_ngram_max: int = 2,
                 tfidf_min_df: int = 3,
                 rf_estimators: int = 300):
        self.model = build_ensemble(random_state, tfidf_max_features, tfidf_ngram_max, tfidf_min_df, rf_estimators)

    def train(self, X, y): # Train model
        logger.info("[*] Training Ensemble")
        self.model.fit(X, y)

    def predict(self, X): # Predict new data
        return self.model.predict(X)

    def evaluate(self, X, y): # Evaluate metrics
        logger.info("[*] Evaluating Ensemble")
        preds = self.predict(X)
        print(classification_report(y, preds, digits=4))

    def save_model(self, path: str): # Save model to joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"[*] Saving ensemble to {path}")
        joblib.dump(self.model, path)

    def load_model(self, path: str): # Load model from joblib
        logger.info(f"[*] Loading ensemble from {path}")
        self.model = joblib.load(path)
