# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from neural import MLP


df = pd.read_csv('data.csv')


np.random.seed(42)
df = df.iloc[np.random.permutation(len(df))]

# Подготовка данных
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1, 1)
X = df.iloc[0:100, [0, 2]].values


X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)

print(f"Исходный диапазон X: [{X.min():.2f}, {X.max():.2f}]")
print(f"Нормализованный диапазон X: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")


inputSize = X_normalized.shape[1]
hiddenSizes = 5
outputSize = 1


iterations = 100
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

print(f"\nНачало обучения: {iterations} итераций, lr={learning_rate}")


# Обучаем сеть
errors = []
for i in range(iterations):
    net.train(X_normalized, y)
    
    predictions = net.predict(X_normalized)
    mse = np.mean(np.square(y - predictions))
    errors.append(mse)
    
    if i % 10 == 0:
        print(f"Итерация: {i:3d} || MSE: {mse:.6f}")

print(f"Финальная ошибка: {errors[-1]:.6f}")

# Оценка на обучающей выборке
pr = net.predict(X_normalized)
pr_binary = (pr > 0.5).astype(int)
errors_count = np.sum(np.abs(y - pr_binary))
accuracy = (len(y) - errors_count) / len(y) * 100

print(f"\nТочность на обучающей выборке: {accuracy:.1f}% ({errors_count} ошибок из {len(y)})")

# Проверка на всей выборке
y_full = df.iloc[:, 4].values
y_full = np.where(y_full == "Iris-setosa", 1, 0).reshape(-1, 1)
X_full = df.iloc[:, [0, 2]].values
X_full_normalized = (X_full - X_min) / (X_max - X_min)

pr_full = net.predict(X_full_normalized)
pr_full_binary = (pr_full > 0.5).astype(int)
errors_full = np.sum(np.abs(y_full - pr_full_binary))
accuracy_full = (len(y_full) - errors_full) / len(y_full) * 100

print(f"Точность на всей выборке: {accuracy_full:.1f}% ({errors_full} ошибок из {len(y_full)})")

print(f"\nПримеры предсказаний (первые 10):")
print(f"Истинные:    {y[:10].flatten()}")
print(f"Предсказано: {pr_binary[:10].flatten()}")
print(f"Вероятности: {pr[:10].flatten().round(3)}")