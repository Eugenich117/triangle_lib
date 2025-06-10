import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random


# 1. Определение пользовательского датасета
class SatelliteDataset(Dataset):
    def __init__(self, image_dir, transform_image=None, transform_mask=None):
        self.image_dir = image_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask

        # Поиск файлов с расширениями .tif, .tiff
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir)
                       if img.lower().endswith(('.tif', '.tiff'))]

        # Генерация путей к синтетическим маскам в папке mask
        mask_dir = os.path.join(image_dir, 'mask')
        os.makedirs(mask_dir, exist_ok=True)  # Создаём папку, если её нет
        self.masks = [
            os.path.join(mask_dir, os.path.basename(img).replace('.tif', '_mask.png').replace('.tiff', '_mask.png'))
            for img in os.listdir(image_dir) if img.lower().endswith(('.tif', '.tiff'))]

        # Отладочный вывод
        print(f"Найдено изображений: {len(self.images)}")
        print(f"Найдено масок: {len(self.masks)}")
        if not self.images:
            raise ValueError(f"В директории {image_dir} не найдено файлов с расширениями .tif или .tiff")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Открываем TIFF-изображение и преобразуем в RGB
        image = Image.open(img_path).convert('RGB')

        # Если маски нет, генерируем синтетическую
        if not os.path.exists(mask_path):
            self.generate_synthetic_mask(img_path, mask_path)

        mask = Image.open(mask_path).convert('L')  # Серые маски с значениями 0, 1, 2, 3

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        # Преобразуем маску в long для целевых меток
        mask = mask.long()  # Предполагаем, что маска уже закодирована (0=фон, 1=облака, 2=дороги, 3=дома)
        return image, mask

    def generate_synthetic_mask(self, img_path, mask_path):
        # Создаём пустую маску
        mask = np.zeros((128, 128), dtype=np.uint8)

        # Генерация облаков с случайными смещениями
        for i in range(5):  # Увеличиваем количество
            x = i * 25 + random.randint(-10, 10)
            y = i * 25 + random.randint(-10, 10)
            size = 20
            x = max(0, min(128 - size, x))  # Ограничиваем координаты
            y = max(0, min(128 - size, y))
            mask[x:x + size, y:y + size] = 1  # Класс 1: облака
            print(f"Облако {i}: x={x}, y={y}, size={size}")

        # Генерация дорог с перекрытием
        for i in range(3):  # Увеличиваем количество
            x = i * 40 + random.randint(-10, 10)
            x = max(0, min(128, x))  # Ограничиваем координаты
            mask[x, :] = 2  # Вертикальная линия по всей высоте, класс 2: дороги
            print(f"Дорога {i}: x={x}, y=0 to 128")

        # Генерация домов с перекрытием
        for i in range(6):  # Увеличиваем количество
            x = i * 20 + random.randint(-10, 10)
            y = 60 + random.randint(-20, 20)
            width, height = 20, 20
            x = max(0, min(128 - width, x))
            y = max(0, min(128 - height, y))
            mask[x:x + width, y:y + height] = 3  # Класс 3: дома
            print(f"Дом {i}: x={x}, y={y}, width={width}, height={height}")

        # Нормализация значений для видимости (0, 85, 170, 255)
        mask = mask * 85

        # Отладочный вывод для проверки значений
        unique_values = np.unique(mask)
        print(f"Значения в маске: {unique_values}")

        # Сохранение маски
        mask_pil = Image.fromarray(mask)
        mask_pil.save(mask_path, format='PNG')  # Явное указание формата
        print(f"Сгенерирована маска: {mask_path}")


# 2. Определение модели UNet для мультиклассовой сегментации
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):  # 4 класса
        super(UNet, self).__init__()

        # Энкодер
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Декодер
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # Учитываем 256 каналов после cat
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # Учитываем 128 каналов после cat
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)  # Выход для 4 классов

        self.relu = nn.ReLU()

    def forward(self, x):
        # Энкодер
        e1 = self.relu(self.enc1(x))  # [batch, 64, H, W]
        e2 = self.relu(self.enc2(self.pool(e1)))  # [batch, 128, H/2, W/2]
        e3 = self.relu(self.enc3(self.pool(e2)))  # [batch, 256, H/4, W/4]

        # Декодер
        d1 = self.relu(self.upconv1(e3))  # [batch, 128, H/2, W/2]
        d1 = torch.cat([d1, e2], dim=1)  # [batch, 256, H/2, W/2]
        d1 = self.relu(self.dec1(d1))  # [batch, 128, H/2, W/2]
        d2 = self.relu(self.upconv2(d1))  # [batch, 64, H, W]
        d2 = torch.cat([d2, e1], dim=1)  # [batch, 128, H, W]
        d2 = self.relu(self.dec2(d2))  # [batch, 64, H, W]
        out = self.final(d2)  # [batch, 4, H, W]

        return out  # Без активации, так как используем CrossEntropyLoss


# 3. Гиперпараметры и загрузка данных
transform_image = transforms.Compose([
    transforms.Resize((128, 128)),  # Уменьшаем размер
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize((128, 128)),  # Уменьшаем размер
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze(0))  # Удаляем единичный канал
])

# Путь к данным
image_dir = r"C:\Users\zheny\OneDrive\Рабочий стол\xxxxxx\универ\IT\питон самоучение\penetrator\pythonProject\3band"
dataset = SatelliteDataset(image_dir, transform_image=transform_image, transform_mask=transform_mask)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 4. Инициализация модели, оптимизатора и функции потерь
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Для мультиклассовой сегментации

# 5. Обучение модели
num_epochs = 50  # Увеличиваем эпохи
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        print(f"Эпоха {epoch + 1}, батч {i}, размер images: {images.shape}, размер masks: {masks.shape}")

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)  # Прямое использование масок как меток
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Батч {i} loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f'Эпоха [{epoch + 1}/{num_epochs}], Средние потери: {avg_loss:.4f}')

# 6. Сохранение модели
torch.save(model.state_dict(), "object_recognition.pth")
print("Модель сохранена как object_recognition.pth")


# 7. Визуализация результата
def visualize_prediction(model, image, mask):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        outputs = model(image)
        pred_mask = outputs.cpu().argmax(dim=1).squeeze(0)  # Получаем предсказанный класс
        print(f"Размер outputs: {outputs.shape}, размер pred_mask: {pred_mask.shape}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().squeeze(0).permute(1, 2, 0))
    plt.title('Изображение')
    plt.subplot(1, 3, 2)
    plt.imshow(mask.cpu().squeeze(0), cmap='viridis')  # Истинная маска
    plt.title('Истинная маска')
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='viridis')  # Предсказанная маска
    plt.title('Предсказанная маска')
    plt.show()


# Пример визуализации
image, mask = dataset[0]
visualize_prediction(model, image, mask)