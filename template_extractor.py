import torch
import timm
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image  # explicit import

class Embedder(nn.Module):
    def __init__(self, model_path="resilience_best.pt"):
        super().__init__()
        model = timm.create_model('resnet18', pretrained=False)
        features = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        self.fc = nn.Linear(features, 512)
        self.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.eval()

    def forward(self, x):
        features = self.backbone(x)
        return nn.functional.normalize(self.fc(features))

class TemplateExtractor:
    def __init__(self):
        self.model = Embedder()
        self.transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor()
        ])

    def extract(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)  # Explicitly convert NumPy array to PIL Image
        img_tensor = self.transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze(0).numpy()
        return embedding

    @staticmethod
    def compare(reference, probe):
        dot_product = np.dot(probe, reference)
        dot_product_normalized = dot_product / (np.linalg.norm(probe) * np.linalg.norm(reference))
        return dot_product_normalized
