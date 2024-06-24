import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import csv
import os

#Pathes
file_dir = "path/to/dataset/"
train_file = "filename.npy"
val_file = "filename.npy"

FEATURE_LENGTH = 1024
#UNI = 1024
#Prov-GigaPath = 1536


csv_file = train_file + '_training_metrics.csv'
csv_header = ['epoch', 'train_loss', 'train_accuracy', 'val_accuracy', 'val_class_0_accuracy', 'val_class_1_accuracy']
# Check if CSV file exists, if not create it and write the header
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

# Load the saved training embeddings and labels
data = np.load(file_dir + train_file + ".npy", allow_pickle=True).item()
embeddings = data['embeddings']
labels = data['labels']
# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Load the validation embeddings and labels
val_data = np.load(file_dir + val_file, allow_pickle=True).item()
val_embeddings = val_data['embeddings']
val_labels = val_data['labels']
# Encode the labels if necessary (assuming label_encoder has been fit on the training labels)
encoded_val_labels = label_encoder.transform(val_labels)

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Create Dataset and DataLoader
dataset = EmbeddingsDataset(embeddings, encoded_labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
# Create Dataset and DataLoader for validation
val_dataset = EmbeddingsDataset(val_embeddings, encoded_val_labels)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


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
#device = torch.device("mps")
model = EmbeddingClassifier(input_size=FEATURE_LENGTH).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # Training
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    train_accuracy = 100. * train_correct / train_total
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Train Accuracy: {train_accuracy}%')

    # Validation
    model.eval()
    class_correct = [0, 0]
    class_total = [0, 0]
    with torch.no_grad():

        val_correct = 0
        val_total = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()

            predicted = (outputs > 0.5).float()
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

            label = int(targets.item())
            class_total[label] += 1
            if predicted.item() == targets.item():
                class_correct[label] += 1

        val_accuracy = 100. * val_correct / val_total
        val_class_0_accuracy = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
        val_class_1_accuracy = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0


        print(
            f'Epoch {epoch + 1}/{num_epochs}, Val Accuracy: {val_accuracy}%, Val Class 0 Accuracy: {val_class_0_accuracy}%, Val Class 1 Accuracy: {val_class_1_accuracy}%')


    # Save metrics to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss / len(train_loader), train_accuracy, val_accuracy, val_class_0_accuracy, val_class_1_accuracy])

    # Save model checkpoint
    if epoch > 8:
        torch.save(model.state_dict(), train_file + f'_checkpoint_epoch_{epoch + 1}.pth')
    else:
        torch.save(model.state_dict(), train_file + f'_checkpoint_epoch_0{epoch + 1}.pth')
