import torch
import torch.nn as nn
import torch.optim as optim
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

t_y = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_y = torch.tensor(t_y).unsqueeze(1)
t_x = torch.tensor(t_x).unsqueeze(1)
t_xn = t_x*0.1

# linear_model = nn.Linear(1,1)
# print("weight: ",linear_model.weight) #tensor([[-0.1335]], requires_grad=True)
# print("bias: ",linear_model.bias) #tensor([-0.4349], requires_grad=True)
# print("params:", list(linear_model.parameters()))

# optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)
# def train_loop(epochs, learning_rate, loss_fn,x, y):
#     for epoch in range(1, epochs + 1):    
#         optimizer.zero_grad()
#         t_p = linear_model(x)
#         loss = loss_fn(y, t_p)
#         loss.backward()
#         optimizer.step()
#         print(f'Epoch: {epoch}, Loss: {float(loss)}')

# train_loop(3000, 1e-2, nn.MSELoss(),t_xn, t_y)
# print("params:", list(linear_model.parameters()))

seq_model = nn.Sequential(
    nn.Linear(1,13),
    nn.Tanh(),
    nn.Linear(13,1))


optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)
def train_loop(epochs, learning_rate, loss_fn,x, y):
    for epoch in range(1, epochs + 1):    
        optimizer.zero_grad()
        t_p = seq_model(x)
        loss = loss_fn(y, t_p)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {float(loss)}')

train_loop(5000, 1e-3, nn.MSELoss(),t_xn, t_y)
print("params:", list(seq_model.parameters()))

from matplotlib import pyplot as plt
t_range = torch.arange(20., 90.).unsqueeze(1)
# fig = plt.figure(dpi=300)
plt.plot(t_x.numpy(), t_y.numpy(),'o')
plt.plot(t_range.numpy(), seq_model(0.1*t_range).detach().numpy(),'c-')
plt.plot(t_x.numpy(), seq_model(t_xn).detach().numpy(),'kx')
plt.show()

#save model
script_model = torch.jit.trace(seq_model, t_xn)
script_model.save("model.pt")