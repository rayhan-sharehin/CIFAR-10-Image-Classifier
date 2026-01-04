# app.py
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io

# CNN Architecture (matches Kaggle trained model)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 convolution layers as in Kaggle notebook
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Flattened size matches trained model
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # 64*16*16 = 16384
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load trained model
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# FastAPI App
app = FastAPI()

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

@app.get("/")
def root():
    return {"message": "CIFAR-10 CNN API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)  # add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {
        "predicted_class": CLASS_NAMES[predicted.item()]
    }
