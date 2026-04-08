import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        self.hiddenSize1 = 10  
        self.hiddenSize2 = 5   
        
        # Веса первого скрытого слоя (вход -> слой 1)
        self.Win1 = np.zeros((1 + inputSize, self.hiddenSize1))
        self.Win1[0, :] = np.random.randint(0, 3, size=(self.hiddenSize1))
        self.Win1[1:, :] = np.random.randint(-1, 2, size=(inputSize, self.hiddenSize1))
        
        # Веса второго скрытого слоя (слой 1 -> слой 2)
        self.Win2 = np.zeros((1 + self.hiddenSize1, self.hiddenSize2))
        self.Win2[0, :] = np.random.randint(0, 3, size=(self.hiddenSize2))
        self.Win2[1:, :] = np.random.randint(-1, 2, size=(self.hiddenSize1, self.hiddenSize2))
        
        # Веса выходного слоя (слой 2 -> выход)
        self.Wout = np.random.randint(0, 2, size=(1 + self.hiddenSize2, outputSize))    .astype(np.float64)
            
    def predict(self, Xp):
            # Первый скрытый слой (вход -> hidden1)
            h1 = np.where((np.dot(Xp, self.Win1[1:, :]) + self.Win1[0, :]) >= 0.0, 1, -1).astype(np.float64)
            
            # Второй скрытый слой (hidden1 -> hidden2)
            h2 = np.where((np.dot(h1, self.Win2[1:, :]) + self.Win2[0, :]) >= 0.0, 1, -1).astype(np.float64)
            
            # Выходной слой (hidden2 -> output)
            out = np.where((np.dot(h2, self.Wout[1:, :]) + self.Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
            
            return out, h2  # Возвращаем результат и последний скрытый слой для обучения
    
    def train(self, X, y, n_iter=5, eta=0.01):
        for i in range(n_iter):
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden2 = self.predict(xi)
                # Обучаем только веса выходного слоя
                self.Wout[1:] += ((eta * (target - pr)) * hidden2).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        return self
