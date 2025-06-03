import torch
import numpy as np
from torch import nn

# === Параметры ===
input_size = 20
hidden_sizes = [256, 128, 64, 32]
num_classes = 5

# Названия режимов телеметрии или типов ошибок
telemetry_labels = {
    0: "Нормальный режим",
    1: "Перегрев двигателя",
    2: "Сбой навигации",
    3: "Разгерметизация отсека",
    4: "Аномальное ускорение"
}

# === Архитектура должна совпадать с обученной ===
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

# === Загрузка модели ===
model = DeepAeroNet()
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# === Пример телеметрии с возможной ошибкой ===
def generate_test_telemetry():
    # Можно заменить на реальные значения
    return np.random.uniform(low=-2, high=2, size=input_size)

sample = generate_test_telemetry()
sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

# === Предсказание ===
with torch.no_grad():
    output = model(sample_tensor)
    predicted_idx = torch.argmax(output).item()
    confidence = torch.softmax(output, dim=1)[0][predicted_idx].item()

# === Результат ===
print(f"🚨 Обнаружен режим: {telemetry_labels[predicted_idx]} (уверенность: {confidence:.2%})")
