import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import os
import csv


# =========================
# MODEL
# =========================
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


# =========================
# LOAD CSV
# =========================
def get_users_from_csv(csv_file):
    users = []
    if not os.path.exists(csv_file):
        print("❌ ไม่พบ CSV")
        return users

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            users.append(row[0])

    return users


# =========================
# TRAIN FUNCTION
# =========================
def train_model_from_csv(
    csv_file="users.csv",
    dataset_folder="dataset",
    epochs=50,
    batch_size=16,
    save_path="face_model.pth"
):
    print("🚀 เริ่มเทรนโมเดล...")

    # ===== โหลด user =====
    users = get_users_from_csv(csv_file)

    if not users:
        print("❌ ไม่มี user")
        return

    # ===== filter dataset =====
    valid_users = []
    for u in users:
        path = os.path.join(dataset_folder, u)
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            valid_users.append(u)
        else:
            print(f"⚠️ ไม่มี dataset: {u}")

    if not valid_users:
        print("❌ ไม่มี dataset ให้ train")
        return

    # sort ให้ label คงที่เสมอ
    valid_users = sorted(valid_users)

    print("✅ Users:", valid_users)

    # ===== Transform =====
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # ===== โหลด dataset =====
    dataset = datasets.ImageFolder(dataset_folder, transform=transform)

    # ===== สร้าง label mapping =====
    class_to_idx = {name: idx for idx, name in enumerate(valid_users)}

    with open("classes.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=4, ensure_ascii=False)

    print("🧠 Mapping:", class_to_idx)

    # ===== remap dataset =====
    new_samples = []
    for path, label in dataset.samples:
        class_name = dataset.classes[label]

        if class_name in class_to_idx:
            new_label = class_to_idx[class_name]
            new_samples.append((path, new_label))

    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]

    # ===== debug =====
    print("📊 Total samples:", len(dataset))
    print("📊 Max label:", max(dataset.targets))
    print("📊 Num classes:", len(class_to_idx))

    # ===== loader =====
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ===== device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Device:", device)

    # ===== model =====
    model = FaceModel(len(class_to_idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for img, label in loader:
            img, label = img.to(device), label.to(device)

            out = model(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # accuracy
            _, pred = torch.max(out, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f} | Acc {acc:.2f}%")

    # ===== save =====
    torch.save(model.state_dict(), save_path)
    print("💾 Saved model!")

    return model

if __name__ == "__main__":
    train_model_from_csv()