import numpy as np
import random
import torch
import torch.nn as nn


"""
y = w1x1 + w2x2 + w3x3 + w4x4
"""

w_list = np.array([2, 3, 4, 7]) # [4, 1]

x_list = []

for _ in range(100):
    x_sample = np.array(
        [random.randint(0, 10) for _ in range(4)]
    )
    x_list.append(x_sample)

y_list = []
for x_sample in x_list:
    # y_sample = np.dot(w_list, x_sample)
    y_sample = w_list @ x_sample
    y_list.append(y_sample)

print(x_list)
print(y_list)


class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        return self.linear(x)
    
model = MyLinear().to("mps")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

x_input = torch.tensor(x_list, dtype=torch.float32).to(DEVICE)
y_input = torch.tensor(y_list, dtype=torch.float32).to(DEVICE)

model.train()

num_epochs = 300
for epoch in range(num_epochs):
    for i, x in enumerate(x_input):        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y_input[i])

        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

with torch.no_grad():
    model.eval()
    print(model.linear.weight)
