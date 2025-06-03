import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
# === Параметры ===
input_size = 20
num_classes = 5
hidden_sizes = [256, 128, 64, 32]
epochs = 1000
batch_size = 32
patience = 50

# === Генерация структурированных данных ===
def generate_telemetry_data(n_samples=5000, input_size=20, num_classes=5):
    """
    Генерация телеметрических данных с неравномерным распределением классов:
    класс 0 — нормальный режим (например, 70%), остальные — аномалии.
    """
    # Пример весов: нормальный режим — 70%, аномалии (1-4) — по 7.5%
    class_weights = np.array([0.7] + [(1 - 0.7) / (num_classes - 1)] * (num_classes - 1))
    class_labels = np.random.choice(num_classes, size=n_samples, p=class_weights)

    X = []
    y = []

    for label in class_labels:
        if label == 0:
            # Нормальный режим — стабильные параметры
            data = np.random.normal(loc=0, scale=0.5, size=input_size)
        elif label == 1:
            # Зависание — почти одинаковые значения
            val = np.random.uniform(-1, 1)
            data = np.full(input_size, val) + np.random.normal(0, 0.01, input_size)
        elif label == 2:
            # Выход за верхнюю границу
            data = np.random.normal(loc=5, scale=0.5, size=input_size)
        elif label == 3:
            # Высокий шум (возможно перегрузка)
            data = np.random.normal(loc=0, scale=3.0, size=input_size)
        elif label == 4:
            # Постепенное нарастание (например, перегрев)
            start = np.random.uniform(-1, 1)
            data = np.linspace(start, start + np.random.uniform(2, 5), input_size)
            data += np.random.normal(0, 0.1, input_size)

        X.append(data)
        y.append(label)

    return np.array(X), np.array(y)



X, y = generate_telemetry_data(samples_per_class=1000, input_size=input_size, num_classes=num_classes)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# === Модель ===
class DeepAeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.LeakyReLU(),

            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.LeakyReLU(),

            nn.Linear(hidden_sizes[3], num_classes)
        )

    def forward(self, x):
        return self.net(x)

# === Инициализация ===
model = DeepAeroNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# === Обучение с валидацией ===
train_losses = []
val_losses = []
best_val_loss = float('inf')
counter = 0
best_model_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Валидация
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    accuracy = correct / total

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2%}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        best_model_state = model.state_dict()  # Сохраняем лучшую модель
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# === Загрузка лучшей модели ===
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# === Визуализация ===
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.grid(True)
plt.show()


# === Загрузка лучшей модели ===
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, "best_model.pth")
    print("✅ Лучшая модель сохранена в 'best_model.pth'")

model = DeepAeroNet()
model_file = "best_model.pth"
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    print("🔁 Модель загружена из сохранённого файла.")
