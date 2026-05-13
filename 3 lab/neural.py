import numpy as np

class MLP:
    def __init__(self, inputSize, outputSize, learning_rate=0.01, hiddenSizes=5):
        
        self.weights = [
            np.random.uniform(-0.5, 0.5, size=(inputSize + 1, hiddenSizes)),
            np.random.uniform(-0.5, 0.5, size=(hiddenSizes + 1, outputSize))
        ]
        self.learning_rate = learning_rate
        self.layers = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return x * (1 - x)
    
    def add_bias(self, x):
        """Добавляет столбец единиц для bias"""
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
     
    def feed_forward(self, x):
        # Входной слой с bias
        input_ = self.add_bias(x)
        
        # Скрытый слой
        hidden_raw = np.dot(input_, self.weights[0])
        hidden_ = self.sigmoid(hidden_raw)
        
        # Скрытый слой с bias для следующего слоя
        hidden_with_bias = self.add_bias(hidden_)
        
        # Выходной слой
        output_raw = np.dot(hidden_with_bias, self.weights[1])
        output_ = self.sigmoid(output_raw)
        
        # Сохраняем слои
        self.layers = [input_, hidden_, output_]
        self.layers_with_bias = [input_, hidden_with_bias, output_]
        
        return output_
    
    def backward(self, target):
        target = np.atleast_2d(target)
        
        # Ошибка выходного слоя
        output = self.layers[2]
        err = (target - output)
        
      
        delta_output = err * self.derivative_sigmoid(output)
        
        
        hidden_with_bias = self.layers_with_bias[1]
        dw_output = np.dot(hidden_with_bias.T, delta_output)
        self.weights[1] += self.learning_rate * dw_output
        
        
        weights_hidden_to_output = self.weights[1][1:, :]
        err_hidden = np.dot(delta_output, weights_hidden_to_output.T)
        
        
        hidden = self.layers[1]
        delta_hidden = err_hidden * self.derivative_sigmoid(hidden)
        
        
        input_ = self.layers[0]
        dw_hidden = np.dot(input_.T, delta_hidden)
        self.weights[0] += self.learning_rate * dw_hidden
            
    def train(self, x_values, target):
            indices = np.random.permutation(x_values.shape[0])
            for idx in indices:
                x_i = x_values[idx:idx+1]
                t_i = target[idx:idx+1]
                self.feed_forward(x_i)
                self.backward(t_i)
    
    def predict(self, x_values):
        x_values = np.atleast_2d(x_values)
        return self.feed_forward(x_values)