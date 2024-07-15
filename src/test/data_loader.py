import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.util import log
logger = log.init_logger()

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        data = pd.read_csv(self.filepath)
        features = data.drop('is_fraudulent', axis=1)
        labels = data['is_fraudulent']
        return features, labels


def send_message(queue, message):
    """Simulate sending a message by placing it in a queue."""
    queue.append(message)  # Send message by appending to the list

def receive_message(queue):
    """Simulate receiving a message by reading from a queue."""
    if queue:
        return queue.pop(0)  # Receive message by popping from the list
    return None

def setup_and_load_data(data_path, test_size=0.2, random_seed=42):
    # Create an instance of DataLoader and load data
    data_loader = DataLoader(str(data_path))
    features, labels = data_loader.load_data()

    # Preprocessing: Scale continuous data and encode categorical data
    column_trans = ColumnTransformer([
        ('scale', StandardScaler(), ['transaction_amount']),
        ('onehot', OneHotEncoder(), ['transaction_type', 'customer_type'])
    ], remainder='passthrough')

    features = column_trans.fit_transform(features)

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_seed)

    # Define a simple neural network model for binary classification
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, x_train, y_train, x_test, y_test