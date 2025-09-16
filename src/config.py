# Set paths
import os

BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIRECTORY, "data", "MID.json")
PROCESSED_DATA_PATH = os.path.join(BASE_DIRECTORY, "data", "processed.csv")
MODEL_DIRECTORY = os.path.join(BASE_DIRECTORY, "saved_models")
LOGS_DIRECTORY = os.path.join(BASE_DIRECTORY, "logs")
