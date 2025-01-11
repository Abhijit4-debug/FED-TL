import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from cryptography.fernet import Fernet
import os
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_dim=33, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='softmax')  # For multiclass classification
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the data
csv1 = pd.read_csv(r'D:\code\.vscode\codes\Federated Learning\NIDS\CIC_IDS-2018\Multi_setting2\setting2_main.csv', low_memory=False)
csv2 = pd.read_csv(r'D:\code\.vscode\codes\Federated Learning\NIDS\CIC_IDS-2018\Multi_setting2\cic_new-2_multi.csv', low_memory=False)

x_train = csv2.drop(columns=['label'], axis=1)
y_train = csv2['label']
x_test = csv1.drop(columns=["Label"], axis=1)
y_test = csv1["Label"]

# Initialize global variables
global_model_parameters = model.get_weights()
class_performance = {}
difference_tracker = {}
threshold = 10
N = 3  # Period after which to check difference threshold

# Encryption key generation and setup
encryption_key = Fernet.generate_key()
cipher = Fernet(encryption_key)


def compute_class_metrics(y_true, y_pred):
    """Compute metrics for each class."""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    return class_accuracy

def save_worst_label_to_csv(round_number, worst_label):
    """Save the round number and worst-performing label to a CSV file."""
    file_name = "new_points_across_rounds.csv"
    data = {"Round": [round_number], "Worst_Label": [worst_label]}
    df = pd.DataFrame(data)

    # Check if the file already exists
    if os.path.exists(file_name):
        # Append without writing the header
        df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        # Write with header if file doesn't exist
        df.to_csv(file_name, mode='w', header=True, index=False)


def update_difference_tracker(current_metrics):
    """Update the difference tracker and calculate the sum of differences."""
    global class_performance, difference_tracker
    differences = {}

    for class_id, metric in enumerate(current_metrics):
        previous_metric = class_performance.get(class_id, 0)
        difference = abs(metric - previous_metric)
        differences[class_id] = difference_tracker.get(class_id, 0) + difference

        # Update the current metrics in the performance tracker
        class_performance[class_id] = metric

    difference_tracker = differences
    return sum(differences.values())


def find_worst_label():
    """Identify the label with the worst performance."""
    worst_label = min(class_performance, key=class_performance.get)
    return worst_label


def encrypt_data(data):
    """Encrypt the data using Fernet encryption."""
    return cipher.encrypt(data.encode())


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return global_model_parameters

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train -----------------")
        model.set_weights(parameters)

        # Train the model
        r = model.fit(x_train, y_train, epochs=10, batch_size=2048, validation_data=(x_test, y_test))
        y_pred_train = np.argmax(model.predict(x_train), axis=1)
        current_metrics = compute_class_metrics(y_train, y_pred_train)

        # Update difference tracker
        total_difference = update_difference_tracker(current_metrics)

        # Check if the difference exceeds the threshold every N rounds
        current_round = config.get("round", 1)
        if current_round % N == 0:
            print(f"Round {current_round}: Total difference = {total_difference}")
            if total_difference > threshold:
                print("Significant performance degradation detected.")
                worst_label = find_worst_label()
                print(f"Worst-performing label: {worst_label}")

                # Encrypt and send the worst label to the server
                encrypted_label = encrypt_data(str(worst_label))
                print(f"Encrypted worst label: {encrypted_label}")

                # Save the worst label to a CSV file
                save_worst_label_to_csv(current_round, worst_label)

                return model.get_weights(), len(x_train), {encrypted_label}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Testing -----------------")
        model.set_weights(parameters)

        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(x_test), axis=1)

        # Print metrics
        print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        cm_df = pd.DataFrame(cm, 
                             index=['Benign', 'DoS attacks-Hulk', 'Bot', 'DoS attacks-SlowHTTPTest', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection'],
                             columns=['Benign', 'DoS attacks-Hulk', 'Bot', 'DoS attacks-SlowHTTPTest', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt=".1f", cmap="GnBu", linewidth=.5)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.show()

        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address="192.1.2.25",
    client=FlowerClient(),
    grpc_max_message_length=1024 * 1024 * 1024
)
