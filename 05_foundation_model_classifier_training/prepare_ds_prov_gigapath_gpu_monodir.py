import timm
from PIL import Image
from torchvision import transforms
import torch
import os
from tqdm import tqdm
import numpy as np

# Folder paths
dataset_path = "path/to/dataset" #single image files
output_file = "path/to/output_file.npy"


model_cfg = {
  "architecture": "vit_giant_patch14_dinov2",
  "num_classes": 0,
  "num_features": 1536,
  "global_pool": "token",
  "model_args": {
    "img_size": 224,
    "in_chans": 3,
    "patch_size": 16,
    "embed_dim": 1536,
    "depth": 40,
    "num_heads": 24,
    "init_values": 1e-05,
    "mlp_ratio": 5.33334,
    "num_classes": 0
  },
  "pretrained_cfg": {
    "tag": "",
    "custom_load": False,
    "input_size": [
      3,
      224,
      224
    ],
    "fixed_input_size": True,
    "interpolation": "bicubic",
    "crop_pct": 1.0,
    "crop_mode": "center",
    "mean": [
      0.485,
      0.456,
      0.406
    ],
    "std": [
      0.229,
      0.224,
      0.225
    ],
    "num_classes": 0,
    "pool_size": None,
    "first_conv": "patch_embed.proj",
    "classifier": "head",
    "license": "prov-gigapath"
  }
}

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create the model
model = timm.create_model(
    model_cfg['architecture'],
    num_classes=model_cfg['num_classes'],
    in_chans=model_cfg['model_args']['in_chans'],
    img_size=model_cfg['model_args']['img_size'],
    patch_size=model_cfg['model_args']['patch_size'],
    embed_dim=model_cfg['model_args']['embed_dim'],
    depth=model_cfg['model_args']['depth'],
    num_heads=model_cfg['model_args']['num_heads'],
    init_values=model_cfg['model_args']['init_values'],
    mlp_ratio=model_cfg['model_args']['mlp_ratio']
).to(device)  # Move model to GPU

weights_path = 'models/prov_gigapath/pytorch_model.bin'
state_dict = torch.load(weights_path)
model.load_state_dict(state_dict)

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
