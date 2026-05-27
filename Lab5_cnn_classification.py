# -*- coding: utf-8 -*-
"""
Лабораторная работа 5: Классификация изображений с помощью CNN
Датасет: породы кошек (собран самостоятельно)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random

# Фиксируем seed для воспроизводимости
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Сначала определим на каком устройстве будем работать - GPU или CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Путь к датасету (папка рядом со скриптом)
DATA_PATH = './my_dataset/'



data_transforms_simple = transforms.Compose([
    transforms.Resize(140),        
    transforms.CenterCrop(128),    
    transforms.ToTensor()
])

# Создаем датасеты
train_dataset_simple = torchvision.datasets.ImageFolder(
    root=DATA_PATH + 'train',
    transform=data_transforms_simple
)
test_dataset_simple = torchvision.datasets.ImageFolder(
    root=DATA_PATH + 'test',
    transform=data_transforms_simple
)

# Смотрим классы
class_names = train_dataset_simple.classes
num_classes = len(class_names)
print(f"\nКлассы: {class_names}")
print(f"Количество классов: {num_classes}")
print(f"Обучающая выборка: {len(train_dataset_simple)} изображений")
print(f"Тестовая выборка: {len(test_dataset_simple)} изображений")

# Создаем загрузчики данных
batch_size = 10
train_loader_simple = torch.utils.data.DataLoader(
    train_dataset_simple, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0
)
test_loader_simple = torch.utils.data.DataLoader(
    test_dataset_simple, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0
)

# Визуализация примеров из ОБУЧАЮЩЕЙ выборки
print("\n--- Визуализация обучающей выборки ---")
inputs, classes = next(iter(train_loader_simple))
img = torchvision.utils.make_grid(inputs, nrow=5)
img = img.numpy().transpose((1, 2, 0))
plt.figure(figsize=(12, 6))
plt.imshow(img, interpolation='nearest')
plt.title(f"Примеры из train (128x128): {[class_names[c] for c in classes[:5]]}")
plt.axis('off')
plt.show()


# --- Создание простой CNN  ---
class CnNet(nn.Module):
    def __init__(self, num_classes=3):
        nn.Module.__init__(self)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16*16*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# --- Обучение простой сети ---
print("\n--- Обучение простой CNN ---")
net_simple = CnNet(num_classes).to(device)
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net_simple.parameters(), lr=0.01) #град.ст.спк

import time
t = time.time()
num_epochs = 50
save_loss_simple = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader_simple):
        images = images.to(device)
        labels = labels.to(device)

        outputs = net_simple(images)
        loss = lossFn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        save_loss_simple.append(loss.item())

        if i % 10 == 0:
            print(f'Эпоха {epoch}/{num_epochs} | Шаг {i} | Ошибка: {loss.item():.4f}')

print(f"\nВремя обучения: {time.time() - t:.2f} сек")

# График loss
plt.figure()
plt.plot(save_loss_simple)
plt.title('Изменение ошибки (простая CNN 128x128)')
plt.xlabel('Итерация')
plt.ylabel('Loss')
plt.show()

# Точность на тесте
correct_predictions = 0
num_test_samples = len(test_dataset_simple)

with torch.no_grad():
    for images, labels in test_loader_simple:
        images = images.to(device)
        labels = labels.to(device)
        pred = net_simple(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

accuracy_simple = 100 * correct_predictions / num_test_samples
print(f'\nТочность простой CNN: {accuracy_simple:.2f}%')

# Сохранение модели
torch.save(net_simple.state_dict(), 'CnNet_cats_128.ckpt')
print("Модель сохранена: CnNet_cats_128.ckpt")



print("\n--- Визуализация предсказаний простой CNN ---")

all_images_simple = []
all_labels_simple = []
for images, labels in test_loader_simple:
    all_images_simple.append(images)
    all_labels_simple.append(labels)

all_images_simple = torch.cat(all_images_simple, dim=0)
all_labels_simple = torch.cat(all_labels_simple, dim=0)

indices_simple = random.sample(range(len(all_labels_simple)), min(10, len(all_labels_simple)))
inputs_simple = all_images_simple[indices_simple]
classes_simple = all_labels_simple[indices_simple]

pred_simple = net_simple(inputs_simple.to(device))
_, pred_class_simple = torch.max(pred_simple.data, 1)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for idx, (img_tensor, pred_label, true_label) in enumerate(zip(inputs_simple, pred_class_simple, classes_simple)):
    if idx >= 10:
        break
    ax = axes[idx // 5, idx % 5]
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)

    color = 'green' if pred_label == true_label else 'red'
    ax.imshow(img, interpolation='nearest')  # Без размытия!
    ax.set_title(f'Предск: {class_names[pred_label]}\nВерно: {class_names[true_label]}', 
                 color=color, fontsize=9)
    ax.axis('off')

plt.suptitle('Простая CNN 128x128 — Случайные тестовые изображения', fontsize=14)
plt.tight_layout()
plt.show()



print("\n" + "="*60)
print("ЧАСТЬ 2: Transfer Learning с предобученной AlexNet")
print("="*60)

# Преобразования для AlexNet (ImageNet: 224x224 + нормализация)
data_transforms_alexnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Пересоздаем датасеты
train_dataset = torchvision.datasets.ImageFolder(
    root=DATA_PATH + 'train',
    transform=data_transforms_alexnet
)
test_dataset = torchvision.datasets.ImageFolder(
    root=DATA_PATH + 'test',
    transform=data_transforms_alexnet
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0
)

# Загружаем предобученную AlexNet
print("\nЗагрузка AlexNet...")
try:
    net = torchvision.models.alexnet(weights='DEFAULT')
    print(" Веса загружены автоматически")
except Exception as e:
    print(f" Ошибка загрузки: {e}")
    net = torchvision.models.alexnet(weights=None)

print(net)

# Замораживаем веса feature extractor
for param in net.parameters():
    param.requires_grad = False

# Заменяем классификатор: 4096 -> num_classes
new_classifier = net.classifier[:-1]
new_classifier.add_module('fc', nn.Linear(4096, num_classes))
net.classifier = new_classifier
net = net.to(device)

# Проверяем точность ДО обучения
print("\n--- Точность до обучения ---")
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

accuracy_before = 100 * correct_predictions / num_test_samples
print(f'Точность до обучения: {accuracy_before:.2f}%')

# --- Обучение классификатора ---
print("\n--- Обучение классификатора AlexNet ---")
num_epochs = 5
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.classifier.parameters(), lr=0.001)

t = time.time()
save_loss = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = lossFn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        save_loss.append(loss.item())

        if i % 10 == 0:
            print(f'Эпоха {epoch}/{num_epochs} | Шаг {i} | Ошибка: {loss.item():.4f}')

print(f"\nВремя обучения: {time.time() - t:.2f} сек")

# График loss
plt.figure()
plt.plot(save_loss)
plt.title('Изменение ошибки (AlexNet Transfer Learning)')
plt.xlabel('Итерация')
plt.ylabel('Loss')
plt.show()

# Итоговая точность
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

accuracy_alexnet = 100 * correct_predictions / num_test_samples
print(f'\nИтоговая точность AlexNet: {accuracy_alexnet:.2f}%')

# Сохранение модели
torch.save(net.state_dict(), 'AlexNet_cats.ckpt')
print("Модель сохранена: AlexNet_cats.ckpt")




print("\n--- Визуализация предсказаний AlexNet ---")

all_images = []
all_labels = []
for images, labels in test_loader:
    all_images.append(images)
    all_labels.append(labels)

all_images = torch.cat(all_images, dim=0)
all_labels = torch.cat(all_labels, dim=0)

indices = random.sample(range(len(all_labels)), min(10, len(all_labels)))
inputs = all_images[indices]
classes = all_labels[indices]

pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for idx, (img_tensor, pred_label, true_label) in enumerate(zip(inputs, pred_class, classes)):
    if idx >= 10:
        break
    ax = axes[idx // 5, idx % 5]
    img = img_tensor.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    color = 'green' if pred_label == true_label else 'red'
    ax.imshow(img, interpolation='nearest')
    ax.set_title(f'Предсказано: {class_names[pred_label]}\nВерно: {class_names[true_label]}', 
                 color=color, fontsize=9)
    ax.axis('off')

plt.suptitle('AlexNet — Случайные тестовые изображения', fontsize=14)
plt.tight_layout()
plt.show()




print("\n--- Confusion Matrix (AlexNet) ---")
try:
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    all_preds = []
    all_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            pred = net(images)
            _, pred_class = torch.max(pred.data, 1)
            all_preds.extend(pred_class.cpu().numpy())
            all_true.extend(labels.numpy())

    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — AlexNet')
    plt.xlabel('Предсказано')
    plt.ylabel('Верно')
    plt.show()
except ImportError:
    print(" sklearn/seaborn не установлены — Confusion Matrix пропущена")



print("\n" + "="*60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*60)
print(f"Простая CNN (128x128):  {accuracy_simple:.2f}%")
print(f"AlexNet до обучения:    {accuracy_before:.2f}%")
print(f"AlexNet после обучения: {accuracy_alexnet:.2f}%")
print("="*60)