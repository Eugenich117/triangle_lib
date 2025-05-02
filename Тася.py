import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import tensorflow.keras.backend as K

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

    # Входные данные (если нужно, можно расширить добавлением Qk или shape как признака)
    X = np.column_stack((shape_flag, Qk, S, v, rho, L, alpha))
    y = np.column_stack((C_da, C_la))

    return X, y


'''def custom_loss_with_penalty(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    # Наказание за отрицательные значения (умножается на 100 — можно настраивать)
    penalty = K.mean(K.relu(-y_pred)) * 100.0
    return mse + penalty'''

# Генерация данных
X, y = generate_data()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Создание модели
model = Sequential([
    Dense(128, input_dim=X.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(2, activation='linear')
])


# Компиляция модели
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss}, Test MAE: {mae}")

# Предсказания и обратное преобразование
y_pred = model.predict(X_test)
y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred)

# Сохранение модели и нормализаторов
model.save('my_model.keras')  # рекомендуемый способ
joblib.dump(scaler_X, 'scaler_X_cpu.save')
joblib.dump(scaler_y, 'scaler_y_cpu.save')

# Обратное преобразование y_pred и y_test (если ты его ещё не сделал)
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

# Разделим на C_d и C_l
C_d_test, C_l_test = y_test_original[:, 0], y_test_original[:, 1]
C_d_pred, C_l_pred = y_pred_original[:, 0], y_pred_original[:, 1]

# --- График сравнения C_d ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(C_d_test, C_d_pred, alpha=0.5, color='dodgerblue')
plt.plot([C_d_test.min(), C_d_test.max()], [C_d_test.min(), C_d_test.max()], 'r--')
plt.xlabel("Настоящий C_d")
plt.ylabel("Предсказанный C_d")
plt.title("Сравнение C_d (истинный vs предсказанный)")
plt.grid(True)

# --- График сравнения C_l ---
plt.subplot(1, 2, 2)
plt.scatter(C_l_test, C_l_pred, alpha=0.5, color='darkorange')
plt.plot([C_l_test.min(), C_l_test.max()], [C_l_test.min(), C_l_test.max()], 'r--')
plt.xlabel("Настоящий C_l")
plt.ylabel("Предсказанный C_l")
plt.title("Сравнение C_l (истинный vs предсказанный)")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Графики ошибок ---
residuals_Cd = C_d_test - C_d_pred
residuals_Cl = C_l_test - C_l_pred

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(residuals_Cd, bins=50, color='skyblue', edgecolor='black')
plt.title("Распределение ошибок C_d")
plt.xlabel("Ошибка")
plt.ylabel("Количество")

plt.subplot(1, 2, 2)
plt.hist(residuals_Cl, bins=50, color='salmon', edgecolor='black')
plt.title("Распределение ошибок C_l")
plt.xlabel("Ошибка")
plt.ylabel("Количество")

plt.tight_layout()
plt.show()

