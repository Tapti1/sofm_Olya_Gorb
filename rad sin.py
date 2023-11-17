import numpy as np
import matplotlib.pyplot as plt
import math

# Создание датасета
X = np.zeros([66, 2])
for i in range(66):
    X[i][0] = i / 4
    X[i][1] = math.sin(X[i][0])

# Инициализация параметров радиально базисной нейросети
num_neurons = 10  # Количество нейронов в слое 
centers = np.linspace(0, 16, num_neurons)  # Равномерно распределенные центры

# Радиально базисная функция расстояния (гауссово ядро)
def gaussian(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

# Обучение
lr = 0.01  # Скорость обучения
sigma = 1.5  # Параметр сглаживания
weights = np.random.rand(num_neurons)  # Инициализация весов

# Обучение
for epoch in range(100):
    for i in range(X.shape[0]):
        # точка
        point = X[i][0]
        # считаем выходы до центров
        outputs = np.array([gaussian(point, c, sigma) for c in centers])

        # Нахождение выхода для данной точки
        predicted = np.dot(outputs, weights)

        # Вычисление ошибки и коррекция весов
        error = X[i][1] - predicted
        weights += lr * error * outputs
    # Предсказание для всех точек
    predictions = []
    for i in range(X.shape[0]):
        point = X[i][0]
        outputs = np.array([gaussian(point, c, sigma) for c in centers])
        predicted = np.dot(outputs, weights)
        predictions.append(predicted)

    # Визуализация результатов
    plt.scatter(X[:, 0], X[:, 1], label='Синус')
    plt.scatter(X[:, 0], predictions, color='red', label='Предсказанные значения')
    plt.legend()
    plt.show()


# Предсказание для всех точек
predictions = []
for i in range(X.shape[0]):
    point = X[i][0]
    # выходы на первом скрытом слое
    outputs = np.array([gaussian(point, c, sigma) for c in centers])
    # выход нейросети
    predicted = np.dot(outputs, weights)
    predictions.append(predicted)

# Визуализация результатов
plt.scatter(X[:, 0], X[:, 1], label='Синус')
plt.scatter(X[:, 0], predictions, color='red', label='Предсказанные значения')
plt.legend()
plt.show()
