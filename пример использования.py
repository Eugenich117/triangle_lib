from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Загрузка модели и масштабировщиков
model = load_model('my_model.keras')
scaler_X = joblib.load('scaler_X_cpu.save')
scaler_y = joblib.load('scaler_y_cpu.save')

# Пример новых входных данных:
# S, v, rho, L, alpha, Qk (в радианах)

# Конус с углом 20 градусов
new_data_cone = np.array([[1, np.deg2rad(90), 5.0, 2500, 1.2, 2.5, 0.4]])

# Масштабирование и предсказание
new_data_scaled = scaler_X.transform(new_data_cone)
prediction_scaled = model.predict(new_data_scaled)
prediction = scaler_y.inverse_transform(prediction_scaled)

C_da, C_la = prediction[0]
print(f"Предсказанный C_da: {C_da:.4f}, C_la: {C_la:.4f}")


# Сфера (Qk = 0 не влияет)
new_data_sphere = np.array([[0, 0.0, 5.0, 2500, 1.2, 2.5, 0.4]])
# Масштабирование и предсказание
new_data_scaled = scaler_X.transform(new_data_sphere)
prediction_scaled = model.predict(new_data_scaled)
prediction = scaler_y.inverse_transform(prediction_scaled)

C_da, C_la = prediction[0]
print(f"Предсказанный C_da: {C_da:.4f}, C_la: {C_la:.4f}")