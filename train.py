import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import json


# ===== Transform =====
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

classes = dataset.classes
print("Classes:", classes)

with open("classes.json", "w") as f:
    json.dump(classes, f)
    
# ===== Model =====
class FaceModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*28*28,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        return self.net(x)

model = FaceModel(len(classes))

# ===== Train =====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    total_loss = 0
    for img, label in loader:
        out = model(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss {total_loss:.4f}")

torch.save(model.state_dict(), "face_model.pth")
print("Saved model!")