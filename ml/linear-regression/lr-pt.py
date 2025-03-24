import torch
import torch.optim as optim

t_y = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_y = torch.tensor(t_y)
t_x = torch.tensor(t_x)
t_xn = t_x*0.1


def model(t_x,w,b):
    return w*t_x + b

def loss_fn(t_y, t_p):
    squared_diff = (t_p-t_y)**2
    return squared_diff.mean()

def dloss_fn(t_y, t_p):
    return 2*(t_p-t_y)

def dmodel_dw(t_x,w,b):
    return t_x

def dmodel_db(t_x,w,b):
    return 1.0

def grad_fn(t_x,w,b,t_y,t_p):
    dw = dloss_fn(t_y,t_p) * dmodel_dw(t_x,w,b)
    db = dloss_fn(t_y,t_p) * dmodel_db(t_x,w,b)
    return torch.tensor([dw.mean(), db.mean()])

def train(learning_rate,w,b,x,y):
    t_p = model(x, w, b)
    loss = loss_fn(y, t_p)
    print("loss: ",float(loss))
    w = w - learning_rate * grad_fn(t_x,w,b,t_y,t_p)[0]
    b = b - learning_rate * grad_fn(t_x,w,b,t_y,t_p)[1]
    t_p = model(x, w, b)
    loss = loss_fn(y, t_p)
    print("loss: ",float(loss))
    return (w,b)

# train(learning_rate=1e-2, w=torch.tensor(1.0), b = torch.tensor(0.0), x=t_x, y = t_y)

# def train_loop(epochs, learning_rate, params, x, y):
#     for epoch in range(1, epochs + 1):
#         w,b = params
#         t_p = model(x, w, b)
#         loss = loss_fn(y, t_p)
#         grad = grad_fn(x,w,b,y,t_p)
#         params = params - learning_rate * grad
#         print(f'Epoch: {epoch}, Loss: {float(loss)}')
#     return params

# param = train_loop(epochs = 5000, 
# learning_rate = 1e-2, 
# params = torch.tensor([1.0,0.0]), 
# x = t_xn, 
# y = t_y)

# print("w,b",float(param[0]), float(param[1]))

# use autograd
params = torch.tensor([1.0,0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.Adam([params],lr=learning_rate)
def train_loop(epochs, learning_rate, params, x, y):
    for epoch in range(1, epochs + 1):    
        optimizer.zero_grad()
        w,b = params
        t_p = model(x, w, b)
        loss = loss_fn(y, t_p)
        loss.backward()
        optimizer.step()
        params = (params - learning_rate * params.grad).detach().requires_grad_()
        print(f'Epoch: {epoch}, Loss: {float(loss)}')
    return params

param = train_loop(epochs = 4000, 
learning_rate = learning_rate, 
params = params,
x = t_xn, 
y = t_y)
print("w,b",float(param[0]), float(param[1]))