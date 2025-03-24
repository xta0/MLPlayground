import torch
import torch.nn as nn

t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_x = torch.tensor(t_x).unsqueeze(1) #convert t_x to [11x1]
t_xn = t_x*0.1

linear_model = nn.Linear(11,1)
print(linear_model.weight)
print(linear_model.bias)
output = linear_model(t_xn.t())
print(output.shape)