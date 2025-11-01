import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from rich import print


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, input: torch.tensor):
        return self.linear(input)


torch.manual_seed(1)

x_train = torch.FloatTensor([
    [73, 80, 75],
    [93, 88, 93],
    [89, 91, 80],
    [96, 98, 100],
    [73, 66, 70]
])
y_train = torch.FloatTensor([152, 185, 180, 196, 142]).unsqueeze(dim=1)

model = LinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

np_epoches = 5000
for epoch in tqdm(range(np_epoches)):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    # 그래디언트 계산
    optimizer.zero_grad()
    cost.backward()

    optimizer.step()

print(f'model: {model}')
with torch.no_grad():
    new_input = torch.FloatTensor([73, 80, 75]).unsqueeze(dim=0)
    prediction = model(new_input)
    print(f'predicted value for input {new_input.squeeze().tolist()}: {prediction}')