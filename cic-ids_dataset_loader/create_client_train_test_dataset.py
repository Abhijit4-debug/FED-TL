import os
import pandas as pd

# Function to load and merge all CSV files in the directory
def load_and_merge_files(directory_path):
    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
    data_frames = [pd.read_csv(file) for file in all_files]
    merged_data = pd.concat(data_frames, ignore_index=True)
    return merged_data

# Function to filter and sample data by attack type
def sample_data(data, attack_type, counts):
    filtered_data = data[data['Label'] == attack_type]
    if len(filtered_data) < sum(counts):
        raise ValueError(f"Not enough samples for attack type: {attack_type}")
    samples = filtered_data.sample(n=sum(counts), random_state=42)
    split_samples = []
    start_idx = 0
    for count in counts:
        split_samples.append(samples.iloc[start_idx:start_idx + count])
        start_idx += count
    return split_samples

directory_path = ""  

# Attack distributions for each client and test dataset
attack_counts = {
    'Benign': [100000, 100000, 100000, 100000, 100000],
    'Bot': [15000, 15000, 12000, 20000, 4500],
    'DDoS attack-HOIC': [0, 100000, 100000, 100000, 10000],
    'DDoS attacks-LOIC-HTTP': [61000, 0, 70000, 75000, 10000],
    'DoS attacks-Hulk': [55000, 60000, 0, 45000, 7500],
    'DoS attacks-SlowHttpTest': [25000, 30000, 15000, 0, 4500],
}

# Load and merge data
merged_data = load_and_merge_files(directory_path)

# Split data based on attack counts
clients = [pd.DataFrame() for _ in range(4)]
test_data = pd.DataFrame()

for attack_type, counts in attack_counts.items():
    client_samples = sample_data(merged_data, attack_type, counts)
    for i in range(4):
        clients[i] = pd.concat([clients[i], client_samples[i]], ignore_index=True)
    test_data = pd.concat([test_data, client_samples[4]], ignore_index=True)

for i, client_data in enumerate(clients, 1):
    client_data.to_csv(f'client_{i}_data.csv', index=False)

test_data.to_csv('test_data.csv', index=False)

print("Datasets created successfully!")
