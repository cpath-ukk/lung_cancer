import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import csv
import os

# CSV file to save metrics
OUTPUT_DIR = "./"
MODEL_DIR = "./checkpoints/"
FILE_DIR = "path/to/dataset"
TEST_FILE = "dataset.npy"

FEATURE_LENGTH = 1024
#UNI = 1024
#Prov-GigaPath = 1536


label_encoder = LabelEncoder()
# Load the validation embeddings and labels
test_data = np.load(FILE_DIR + TEST_FILE, allow_pickle=True).item()
test_embeddings = test_data['embeddings']
test_labels = test_data['labels']
# Encode the labels if necessary (assuming label_encoder has been fit on the training labels)
encoded_test_labels = label_encoder.fit_transform(test_labels)

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Create Dataset and DataLoader for test
test_dataset = EmbeddingsDataset(test_embeddings, encoded_test_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define a simple neural network model
class EmbeddingClassifier(nn.Module):
    def __init__(self, input_size):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Model, loss function, and optimizer
# Assuming CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingClassifier(input_size=FEATURE_LENGTH).to(device)

checkpoints = sorted(os.listdir(MODEL_DIR))

for checkpoint in checkpoints:
    # Load the state dictionary from the checkpoint file
    checkpoint_path = MODEL_DIR + checkpoint  # Replace with your checkpoint file path
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    csv_file = OUTPUT_DIR + checkpoint + '.csv'
    csv_header = ['GT', checkpoint]
    # Check if CSV file exists, if not create it and write the header
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            outputs_np = outputs.cpu().detach().numpy()
            label = int(targets.item())

            # Save metrics to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([outputs_np, label])
