import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('dataset_simple.csv')
df.columns = df.columns.str.strip()

X_raw = df.iloc[:, [0]].values # Берем 1-й столбец 
y_raw = df.iloc[:, [1]].values # Берем 2-й столбец 


X_mean, X_std = X_raw.mean(), X_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()

X = torch.Tensor((X_raw - X_mean) / X_std)
y = torch.Tensor((y_raw - y_mean) / y_std)

    
class IncomeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(1, 10),  
            nn.ReLU(),         
            nn.Linear(10, 1),   
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layers(x)

net = IncomeNet()


lossFn = nn.MSELoss() 

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


epochs = 1000 
for i in range(epochs):
    pred = net(X)           
    loss = lossFn(pred, y)  

    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()        

    if i % 200 == 0:
        print(f"Эпоха {i}: Ошибка = {loss.item():.4f}")


with torch.no_grad():
    
    pred_real = net(X) * y_std + y_mean
    
    mae = torch.mean(torch.abs(pred_real - torch.Tensor(y_raw)))
    print(f"\nСредняя абсолютная ошибка: {mae.item():.2f}")


plt.figure(figsize=(10, 5))
plt.scatter(X_raw, y_raw, label='Реальные данные', alpha=0.6)

# Рисуем плавную линию предсказания
with torch.no_grad():
    x_range = torch.linspace(X_raw.min(), X_raw.max(), 100).reshape(-1,1)
    y_range = net((x_range - X_mean) / X_std) * y_std + y_mean
    plt.plot(x_range.numpy(), y_range.numpy(), color='red', label='Линия регрессии', linewidth=3)

plt.xlabel("Возраст")
plt.ylabel("Доход")
plt.legend()
plt.title("Вариант №12: Предсказание дохода по возрасту")
plt.show()