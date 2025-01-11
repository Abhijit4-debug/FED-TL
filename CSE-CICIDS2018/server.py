import flwr as fl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from typing import List, Dict, Any

# Global variables
exemplar_set = {}  # Exemplar set P
threshold = 0.0  # Threshold for classifying new/existing classes

# Function to compute mean representation of each exemplar class
def compute_mean_representation(exemplar_set):
    """Calculate mean representation of each exemplar class."""
    mean_representations = {}
    for label, data_points in exemplar_set.items():
        mean_representation = np.mean(data_points, axis=0)
        mean_representations[label] = mean_representation
    return mean_representations

# Function to compute the classification threshold
def compute_threshold(mean_representations):
    """Calculate the classification threshold based on mean representations."""
    global threshold
    labels = list(mean_representations.keys())
    num_classes = len(labels)

    weighted_avg = 0.0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            mu_i = mean_representations[labels[i]]
            mu_j = mean_representations[labels[j]]
            dist = np.linalg.norm(mu_i - mu_j)
            weighted_avg += dist / (num_classes * (num_classes - 1) / 2)

    threshold = weighted_avg
    return threshold

# Function to classify new data points
def classify_data(data_point, mean_representations, threshold):
    """Classify new data points into new or existing classes."""
    distances = {label: np.linalg.norm(data_point - mu) for label, mu in mean_representations.items()}
    min_label = min(distances, key=distances.get)
    if distances[min_label] > threshold:
        return "New Class"
    else:
        return min_label

# Function to train the model on the exemplar set
def train_global_model_on_exemplars(exemplar_set):
    """Train a global model using the exemplar set."""
    global_model = Sequential()
    global_model.add(Dense(20, input_dim=33, activation='relu'))
    global_model.add(Dense(20, activation='relu'))
    global_model.add(Dense(20, activation='softmax'))  # For multiclass classification
    global_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Prepare training data from exemplar set
    X_train = []
    y_train = []
    for label, data_points in exemplar_set.items():
        X_train.extend(data_points)
        y_train.extend([label] * len(data_points))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train the model
    global_model.fit(X_train, y_train, epochs=10, batch_size=2048, verbose=1)
    return global_model.get_weights()

# Strategy extending FedAvg for the described functionality
class CustomFedAvg(fl.server.strategy.FedProx):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = None  # Global model parameters

    def aggregate_fit(
        self, rnd: int, results: List[fl.server.client_proxy.FitRes], failures: List[BaseException]
    ) -> fl.common.Weights:
        """Aggregate fit results and train the global model."""
        global exemplar_set, threshold

        # Call the parent method for aggregation
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            self.global_model = aggregated_weights

        # Train using the exemplar set if necessary
        if rnd % self.config["N"] == 0:  # Frequency of Transfer Learning process
            mean_representations = compute_mean_representation(exemplar_set)
            threshold = compute_threshold(mean_representations)

            new_data_points = pd.read_csv("new_points_across_rounds.csv")
            classification_result = classify_data(new_data_points, mean_representations, threshold)
            print(f"Classification result for new data point: {classification_result}")

            if classification_result == "New Class":
                print("New class detected. Training global model on exemplar set.")
                new_weights = train_global_model_on_exemplars(exemplar_set)
                self.global_model = new_weights

        return self.global_model

    def on_fit_config_fn(self, rnd: int):
        """Send config to clients, including exemplar data if applicable."""
        global exemplar_set

        for label in exemplar_set:
            np.random.shuffle(exemplar_set[label])

        return {"exemplar_set": exemplar_set, "round": rnd}

# Load data from CSV
csv_data = pd.read_csv("friday_3.csv")
X = csv_data.drop(columns=["label"])
y = csv_data["label"]

# Split data into train and exemplar set
X_train, X_exemplar, y_train, y_exemplar = train_test_split(X, y, test_size=0.2, stratify=y)

# Populate the exemplar set
for label in np.unique(y_exemplar):
    exemplar_set[label] = X_exemplar[y_exemplar == label].values

# Define the strategy
strategy = CustomFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    proximal_mu=1,
    N=5  # Frequency of Transfer Learning process
)

# Start the Flower server
fl.server.start_server(
    server_address="192.1.2.25",
    config=fl.server.ServerConfig(num_rounds=10),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)
