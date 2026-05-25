# Configuration for the Health Monitoring Project

# File paths
DATA_PATH = '../data/smart_healthcare_dataset(1).csv'
PROCESSED_DATA_DIR = '../data/processed/'

# Features to drop during training
DROP_COLUMNS = ['heart_disease', 'diabetes', 'stroke', 'health_risk_score']

# Target columns used to create the 'Bad Health' label
TARGET_CONSTITUENTS = ['heart_disease', 'diabetes', 'stroke']

# Hyperparameters for Data Loading
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42