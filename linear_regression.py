import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from rich import print

torch.manual_seed(1)

x_train = torch.FloatTensor([
    [73, 80, 75],
    [93, 88, 93],
    [89, 91, 80],
    [96, 98, 100],
    [73, 66, 70]
])
y_train = torch.FloatTensor([152, 185, 180, 196, 142]).unsqueeze(dim=1)

W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = lambda x: (x @ W) + b
loss = lambda x, y: torch.mean((x - y) ** 2)

optimizer = optim.SGD([W, b], lr=1e-5)

nb_epoches = 5000
for epoch in tqdm(range(nb_epoches)):
    cost = loss(hypothesis(x_train), y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print('[red]Linear Regression Model[/red]')
print(f'[orange]W: {W}\nb: {b}[/orange]')

with torch.no_grad():
    new_input = torch.FloatTensor([73, 66, 70]).unsqueeze(dim=0)
    prediction = hypothesis(new_input)
    print(f'predicted value for input {new_input.squeeze().tolist()}: {prediction}')