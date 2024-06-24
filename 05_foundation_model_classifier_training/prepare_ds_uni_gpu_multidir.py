import timm
from PIL import Image
from torchvision import transforms
import torch
import os
from tqdm import tqdm
import numpy as np

# Folder paths
NAME = "dataset_name"

dataset_path = "path/to/dataset/folder" + NAME +  "/test/"
output_file = "path/to/output/dir" + NAME + "/" + NAME + ".npy"

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
).to(device)  # Move model to GPU

model.load_state_dict(torch.load(os.path.join("models/uni", "pytorch_model.bin")), strict=True)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

model.eval()

def get_embedding(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    sample_input = transform(image).unsqueeze(0).to(device)  # Move input to GPU
    with torch.no_grad():
        output = model(sample_input).squeeze()
    return output.cpu().numpy()

# Initialize a list to store embeddings and corresponding labels
embeddings = []
labels = []

# Iterate over the dataset directory structure
print("Dataset: ", dataset_path)
print("Target: ", output_file)

# Iterate over the dataset directory structure
for class_dir in sorted(os.listdir(dataset_path)):
    print("Start for class: ", class_dir)
    class_path = os.path.join(dataset_path, class_dir)
    if os.path.isdir(class_path):
        for image_name in tqdm(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_name)
            if image_path.endswith(".jpg") or image_path.endswith(".png"):
                embedding = get_embedding(model, image_path, device)
                embeddings.append(embedding)
                labels.append(class_dir)

# Convert embeddings and labels to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Save embeddings and labels to a file
np.save(output_file, {"embeddings": embeddings, "labels": labels})
