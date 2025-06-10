import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Генерация синтетических данных
def generate_data(num_samples=100_000, shape='cone', Qk_input=None):
    S = np.random.uniform(1, 10, num_samples)
    v = np.random.uniform(100, 10000, num_samples)
    rho = np.random.uniform(0.02, 1.225, num_samples)
    L = np.random.uniform(0.2, 10, num_samples)
    alpha = np.random.uniform(0, np.pi / 2, num_samples)
    r1 = np.random.uniform(0.1, 0.5, num_samples)
    r2 = np.random.uniform(0.3, 1.0, num_samples)
    theta = np.zeros(num_samples)  # Угол атаки 0
    shape_flag = np.ones(num_samples) if shape == 'cone' else np.zeros(num_samples)

    # Конусный угол
    if Qk_input is not None:
        Qk = np.full(num_samples, Qk_input)
    else:
        Qk = np.random.uniform(0, np.pi / 4, num_samples)

    if shape == 'sphere':
        # Константные значения для сферы
        C_d = np.full(num_samples, 0.47)  # Приближённое значение для сферы
        C_l = np.zeros(num_samples)
    else:
        # Формулы для конуса
        C_d = ((2 * L * r2 * (1 + r1 / r2)) / S) * (np.tan(Qk) / 2) * (2 * np.cos(theta) ** 2 * np.sin(Qk) ** 2 + np.sin(theta))
        C_l = ((2 * L * r2 * (1 + r1 / r2)) / S) * np.pi * np.cos(theta) * np.sin(theta) * np.cos(Qk) ** 2

    # Проекция коэффициентов
    C_da = C_d * np.cos(alpha) + C_l * np.sin(alpha)
    C_la = C_d * np.sin(alpha) + C_l * np.cos(alpha)

    # Конвективный тепловой поток (упрощённая версия)
    initial['ro'] = rho
    initial['V'] = v
    initial['qk'] = 0
    equations = [qk_func]
    dt = 0.1
    values = euler(equations, initial, dt)
    q_k = values[0]
    T_w = ((q_k / (0.8 * 5.67 * 10**(-8)))**0.25)  # Равновесная температура
    # Входные данные
    X = np.column_stack((shape_flag, Qk, S, v, rho, L, alpha))
    # Выходные данные: C_da, C_la, T_w, q_k
    y = np.column_stack((C_da, C_la, T_w, q_k))

    return X, y

def qk_func(initial):
    ro = initial['ro']
    V = initial['V']
    dqk = ((1.318 * 10 ** 5) / np.sqrt(0.5)) * ((ro / 1.2255) ** 0.5) * ((V / 7910) ** 3.25)
    return dqk, 'qk'

def euler(equations, initial, dt):
    # в equations пишем названия функций с уравнениями, а в initial пишем все переменные, которые нам нужны
    new_value_list = [0] * len(equations)

    for i, eq in enumerate(equations):
        derivative, key = eq(initial)
        new_value_list[i] = initial[key] + derivative * dt  # Обновляем значение переменной по индексу

    return new_value_list

# Нейронная сеть на PyTorch
class AeroThermalNet(nn.Module):
    def __init__(self, input_dim=7, hidden_dims=[128, 128, 64], output_dim=4):
        super(AeroThermalNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(0.3))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.Dropout(0.3))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
initial = {}
# Генерация и подготовка данных
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Преобразование в тензоры PyTorch
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Инициализация модели, оптимизатора и функции потерь
model = AeroThermalNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Обучение с Early Stopping
early_stopping_patience = 30
best_loss = float('inf')
patience_counter = 0

num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Проверка валидации
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_train_tensor)  # Используем часть обучающих данных для валидации
        val_loss = criterion(val_outputs, y_train_tensor)

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break

# Загрузка лучшей модели
model.load_state_dict(torch.load('best_model.pth'))

# Оценка на тестовых данных
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.cpu().numpy()
    y_test_original = scaler_y.inverse_transform(y_test)
    y_pred_original = scaler_y.inverse_transform(y_pred)

# Разделение предсказаний
C_da_test, C_la_test, T_w_test, q_k_test = y_test_original[:, 0], y_test_original[:, 1], y_test_original[:, 2], y_test_original[:, 3]
C_da_pred, C_la_pred, T_w_pred, q_k_pred = y_pred_original[:, 0], y_pred_original[:, 1], y_pred_original[:, 2], y_pred_original[:, 3]

# Графики сравнения
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(C_da_test, C_da_pred, alpha=0.5, color='dodgerblue')
plt.plot([C_da_test.min(), C_da_test.max()], [C_da_test.min(), C_da_test.max()], 'r--')
plt.xlabel("Настоящий C_da")
plt.ylabel("Предсказанный C_da")
plt.title("Сравнение C_da")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.scatter(C_la_test, C_la_pred, alpha=0.5, color='darkorange')
plt.plot([C_la_test.min(), C_la_test.max()], [C_la_test.min(), C_la_test.max()], 'r--')
plt.xlabel("Настоящий C_la")
plt.ylabel("Предсказанный C_la")
plt.title("Сравнение C_la")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.scatter(T_w_test, T_w_pred, alpha=0.5, color='green')
plt.plot([T_w_test.min(), T_w_test.max()], [T_w_test.min(), T_w_test.max()], 'r--')
plt.xlabel("Настоящий T_w")
plt.ylabel("Предсказанный T_w")
plt.title("Сравнение T_w")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.scatter(q_k_test, q_k_pred, alpha=0.5, color='red')
plt.plot([q_k_test.min(), q_k_test.max()], [q_k_test.min(), q_k_test.max()], 'r--')
plt.xlabel("Настоящий q_k")
plt.ylabel("Предсказанный q_k")
plt.title("Сравнение q_k")
plt.grid(True)

plt.tight_layout()
plt.show()

# Сохранение модели и нормализаторов
torch.save(model.state_dict(), 'aero_thermal_model.pth')
joblib.dump(scaler_X, 'scaler_X.pth')
joblib.dump(scaler_y, 'scaler_y.pth')