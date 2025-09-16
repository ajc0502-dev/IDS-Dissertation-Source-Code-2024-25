# Import Modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
from src.logger import logger

class DTIDS: # DecisionTreeClassifier
    def __init__(self, max_depth=5, random_state=42): # Shallow max depth and reproducable random state
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def train(self, X, y): # Fixes DT Trainer
        logger.info("[*] Training Decision Tree") # Logs to serial
        self.model.fit(X, y)

    def predict(self, X): # Rewturn predictions 
        return self.model.predict(X)

    def evaluate(self, X, y): # Produce classification report
        logger.info("[*] Evaluating Decision Tree")
        preds = self.model.predict(X)
        print(classification_report(y, preds, digits=4)) # Report desired metrics

    def save_model(self, path: str):
        logger.info(f"[*] Saving model to {path}") # Saves fitted model
        joblib.dump(self.model, path)

    def load_model(self, path: str): # Loads fitted model
        logger.info(f"[*] Loading model from {path}")
        self.model = joblib.load(path)
