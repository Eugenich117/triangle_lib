import torch
import joblib
import numpy as np
import math as m

# Определение архитектуры модели (должна совпадать с обучающей версией)
class AeroThermalNet(torch.nn.Module):
    def __init__(self, input_dim=7, hidden_dims=[128, 128, 64], output_dim=4):
        super(AeroThermalNet, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm1d(hidden_dims[0]))
        layers.append(torch.nn.Dropout(0.3))

        for i in range(len(hidden_dims) - 1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(torch.nn.Dropout(0.3))

        layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка модели и масштабировщиков
model = AeroThermalNet().to(device)
model.load_state_dict(torch.load('aero_thermal_model.pth'))
model.eval()  # Перевод модели в режим оценки

scaler_X = joblib.load('scaler_X.pth')
scaler_y = joblib.load('scaler_y.pth')

# Пример новых входных данных:
# S, v, rho, L, alpha, Qk (в радианах), shape_flag (1 для конуса, 0 для сферы)
d = 0.8
S = (m.pi * d ** 2) / 4
# Конус с углом 20 градусов
new_data_cone = np.array([[1, np.deg2rad(20), S, 2500, 0.9, 2.5, 0.4]])  # shape_flag=1 для конуса

# Масштабирование и предсказание
new_data_scaled = scaler_X.transform(new_data_cone)
new_data_tensor = torch.FloatTensor(new_data_scaled).to(device)
with torch.no_grad():
    prediction_scaled = model(new_data_tensor)
    prediction = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())

C_da, C_la, T_w, q_k = prediction[0]
print(f"Предсказанный C_da: {C_da:.4f}, C_la: {C_la:.4f}, T_w: {T_w:.4f}, q_k: {q_k:.4f}")

# Сфера (Qk = 0 не влияет, shape_flag=0)
new_data_sphere = np.array([[0, 0.0, S, 2500, 1.2, 2.5, 0.4]])  # shape_flag=0 для сферы

# Масштабирование и предсказание
new_data_scaled = scaler_X.transform(new_data_sphere)
new_data_tensor = torch.FloatTensor(new_data_scaled).to(device)
with torch.no_grad():
    prediction_scaled = model(new_data_tensor)
    prediction = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())

C_da, C_la, T_w, q_k = prediction[0]
print(f"Предсказанный C_da: {C_da:.4f}, C_la: {C_la:.4f}, T_w: {T_w:.4f}, q_k: {q_k:.4f}")