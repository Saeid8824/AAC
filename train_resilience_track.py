# -*- coding: utf-8 -*-
import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses
import umap.umap_ as umap
import matplotlib.pyplot as plt
from PIL import Image
import timm
from config_runtime import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvPairDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform

        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            if not folder.startswith("pair_") or "_label_" not in folder:
                continue

            try:
                label = int(folder.split("_label_")[-1])
                idx = int(folder.split("_")[1])
            except Exception:
                continue

            files = os.listdir(folder_path)
            im0_path = None
            im1_path = None

            for f in files:
                if f.lower().startswith("im_0"):
                    im0_path = os.path.join(folder_path, f)
                elif f.lower().startswith("im_1"):
                    im1_path = os.path.join(folder_path, f)

            if im0_path and im1_path:
                self.samples.append((im0_path, im1_path, label, idx))
            else:
                print(f"Skipping {folder}: could not find both im_0 and im_1")

        print(f"Loaded {len(self.samples)} valid image pairs from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        im0_path, im1_path, label, pair_idx = self.samples[idx]
        im0 = self.transform(Image.open(im0_path).convert("RGB"))
        im1 = self.transform(Image.open(im1_path).convert("RGB"))
        return im0, im1, torch.tensor(label, dtype=torch.float32), pair_idx


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model(config['model_type'], pretrained=True)

        if hasattr(model, 'fc'):
            features = model.fc.in_features
            model.fc = nn.Identity()
        elif hasattr(model, 'head'):
            features = model.head.in_features
            model.head = nn.Identity()
        else:
            raise ValueError("Unsupported model type")

        self.backbone = model
        self.fc = nn.Linear(features, 512)

    def forward(self, x):
        features = self.backbone(x)
        return nn.functional.normalize(self.fc(features))


def get_transform():
    aug = []
    if config.get('horizontal_flip'): aug.append(transforms.RandomHorizontalFlip())
    if config.get('vertical_flip'): aug.append(transforms.RandomVerticalFlip())
    if config.get('rotation'): aug.append(transforms.RandomRotation(15))
    aug.extend([transforms.Resize((500, 500)), transforms.ToTensor()])
    return transforms.Compose(aug)


def cosine_similarity_loss(emb1, emb2, labels, margin=0.5):
    cos_sim = nn.functional.cosine_similarity(emb1, emb2)
    pos_loss = (1 - cos_sim) * labels
    neg_loss = torch.clamp(cos_sim - margin, min=0.0) * (1 - labels)
    return (pos_loss + neg_loss).mean()


def train_resilience():
    dataset_dir = config['resilience_data_dir']
    transform = get_transform()
    dataset = AdvPairDataset(dataset_dir, transform)
    if len(dataset) == 0:
        raise RuntimeError("No valid data loaded. Check dataset directory or filenames.")

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = Embedder().to(device)
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['lr'])
    scheduler = getattr(optim.lr_scheduler, config['scheduler'])(optimizer, T_max=config['n_epochs'])

    logs = {}

    for epoch in range(500):	#(config['n_epochs']):
        model.train()
        epoch_losses = []

        for im0, im1, labels, _ in loader:
            im0, im1, labels = im0.to(device), im1.to(device), labels.to(device)

            optimizer.zero_grad()
            emb0 = model(im0)
            emb1 = model(im1)
            loss = cosine_similarity_loss(emb0, emb1, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        logs[epoch] = {"loss": avg_loss}
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "resilience_best.pt")
        scheduler.step()

    with open("resilience_train_log.json", "w") as f:
        json.dump(logs, f, indent=2)

    visualize_embeddings(model, loader)


def visualize_embeddings(model, loader):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for im0, im1, label, _ in loader:
            for im in [im0, im1]:
                emb = model(im.to(device)).cpu().numpy()
                embeddings.append(emb)
                labels.extend(label.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    reducer = umap.UMAP(min_dist=config.get("umap_dist", 0.1))
    proj = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='jet', alpha=0.7)
    plt.colorbar(label="Label (0 = impersonation, 1 = evasion)")
    plt.title("UMAP Projection of Resilience Embeddings")
    plt.show()


if __name__ == "__main__":
    train_resilience()
