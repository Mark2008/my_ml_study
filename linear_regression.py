import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from rich import print

torch.manual_seed(1)

x_train = torch.FloatTensor([1,2,3]).unsqueeze(dim=1)
y_train = torch.FloatTensor([3,5,7]).unsqueeze(dim=1)


W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = lambda x: W * x + b
loss = lambda x, y: torch.mean((x - y) ** 2)

cost = loss(hypothesis(x_train), y_train)
optimizer = optim.SGD([W, b], lr=0.01)

nb_epoches = 5000
for epoch in tqdm(range(nb_epoches)):
    cost = loss(hypothesis(x_train), y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print('[red]Linear Regression Model[/red]')
print(f'[orange]W: {W}\nb: {b}[/orange]')