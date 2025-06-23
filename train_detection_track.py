import os
import torch
import timm
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

# CONFIGURATION
root_dir = "./AdvCelebA"
image_dir = os.path.join(root_dir, "images")
attack_files_dir = root_dir

model_type = "resnet18"
batch_size = 64
lr = 1e-4
n_epochs = 100

# 1. Collect all adversarial image names from attack text files
adversarial_images = set()
for fname in os.listdir(attack_files_dir):
    if fname.startswith("final_attack_attackid_") and fname.endswith(".txt"):
        with open(os.path.join(attack_files_dir, fname), "r") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split()
                if parts:
                    adversarial_images.add(parts[0])  # attacked_image

# 2. Custom dataset using image_dir and adversarial set
class DetectionDataset(Dataset):
    def __init__(self, image_dir, adversarial_set, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.adversarial_set = adversarial_set
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, fname)
        label = 1 if fname in self.adversarial_set else 0
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# 3. Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 4. Dataset and DataLoader
dataset = DetectionDataset(image_dir, adversarial_images, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Model
class DetectionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model(model_type, pretrained=True)
        if hasattr(model, 'fc'):
            features = model.fc.in_features
            model.fc = nn.Identity()
        elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            features = model.head.in_features
            model.head = nn.Identity()
        else:
            raise ValueError("Cannot determine feature dimension from model.")
        self.backbone = model
        self.classifier = nn.Linear(features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return torch.sigmoid(self.classifier(features))

# 6. Training loop
model = DetectionClassifier().cuda()
opt = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
loss_fn = nn.BCELoss()

for epoch in range(n_epochs):
    model.train()
    losses = []
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.cuda()
        labels = labels.unsqueeze(1).cuda()
        opt.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, labels)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    avg_loss = np.mean(losses)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0.0
    f1 = f1_score(y_true, (np.array(y_pred) > 0.5).astype(int))
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
    scheduler.step()

torch.save(model.state_dict(), "detection_model_final.pt")
