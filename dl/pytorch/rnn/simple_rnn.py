import torch
from torch import nn
import numpy as np

seq_len = 20

time_steps = np.linspace(0, np.pi, seq_len + 1) # (21,)
data = np.sin(time_steps)  # (21,)
data.resize((seq_len+1, 1)) # (21, 1)

x = data[:-1] # all but the last piece of data
y = data[1:] # all but the first, y is one step ahead of x

print("x: ", x.shape) #(20, 1) (0,1,2,3,...,19)
print("y: ", y.shape) #(20, 1) (1,2,3,4,...,20) 

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)
        return output, hidden

# test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)

# time_steps = np.linspace(0, np.pi, seq_len)
# data = np.sin(time_steps)
# data.resize((seq_len, 1))

# test_input = torch.Tensor(data).unsqueeze(0)
# print('Input size: ', test_input.size()) # [1, 20, 1]) (batch_size, seq_len, input_size (#sequence))
# test_out, test_h = test_rnn(test_input, None)

# print('Output size: ', test_out.size()) # [20, 1]
# print('Hidden State size: ', test_h.size()) #[2, 1, 10] (#layers, batch_size, hidden_size)

input_size = 1
output_size  = 1
hidden_dim = 32
n_layers = 1

rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

 # This is a regression problem: can we train an RNN to accurately predict the next data point
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

def train(rnn, n_steps, print_every):
    hidden = None
    for batch_i, step in enumerate(range(n_steps)):
        time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_len+1)
        data = np.sin(time_steps)
        data.resize((seq_len+1, 1))
        x = data[:-1]
        y = data[1:]
        x_tensor = torch.Tensor(x).unsqueeze(0)
        y_tensor = torch.Tensor(y)
        
        prediction, hidden = rnn(x_tensor, hidden)
        
        hidden = hidden.data
        
        loss = criterion(prediction, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_i % print_every == 0:
            print("Loss: ", loss.item())
            
    return rnn

n_steps = 75
print_every = 15
trained_rnn = train(rnn, n_steps, print_every)
        