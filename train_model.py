import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. ตั้งค่าพารามิเตอร์และพาธโฟลเดอร์
# ==========================================
# ชี้ไปที่โฟลเดอร์ emotions ของคุณ
TRAIN_DIR = "./emotions/train"
VAL_DIR = "./emotions/test"

BATCH_SIZE = 32
NUM_EPOCHS = 10              # ปรับเพิ่มได้ตามต้องการ เช่น 15 หรือ 20
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "custom_emotion_model.pth"

# เช็กว่ามี GPU หรือไม่
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"กำลังใช้งาน Device: {device}")

# ==========================================
# 2. การเตรียมรูปภาพ (Data Augmentation)
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# โหลด Datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms['val'])

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = train_dataset.classes
NUM_CLASSES = len(class_names)

print(f"พบทั้งสิ้น {NUM_CLASSES} คลาส: {class_names}")
print(f"จำนวนภาพ Train: {dataset_sizes['train']} ภาพ | Validation: {dataset_sizes['val']} ภาพ")

# ==========================================
# 3. โหลดและปรับแต่ง MobileNetV3
# ==========================================
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# ปรับ Output Layer ให้เป็น 7 คลาสตามชุดข้อมูล
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. ฟังก์ชันเริ่มการเทรน
# ==========================================
def train_model():
    since = time.time()
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # บันทึก Weight ที่ดีที่สุดลงไฟล์
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"--> [Saved] บันทึกโมเดลใหม่ที่ความแม่นยำ: {best_acc*100:.2f}%")

    time_elapsed = time.time() - since
    print(f"\nเทรนเสร็จสิ้นภายใน {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"ความแม่นยำสูงสุด (Validation Accuracy): {best_acc*100:.2f}%")

if __name__ == '__main__':
    train_model()