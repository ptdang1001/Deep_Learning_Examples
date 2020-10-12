import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PI = 3.14


# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()
# sys.exit()


# this is a Network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # ReLu function for hidden layer
        x = self.predict(x)  # linear output
        return x


n_feature = 1  # the number of input neurons
n_hidden = 1000  # the number of hidden neurons
n_output = 1  # the number of output neurons
net = Net(n_feature, n_hidden, n_output)  # define the network

print(net)  # net architecture

# define the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
#define the loss function to MSELoss
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

#plt.ion()  # something about plotting

x= torch.unsqueeze(torch.linspace(0, 24, 100),
                    dim=1)  # x data (tensor), shape=(100, 1)
y = 100 + torch.cos(PI * (torch.true_divide(x, 12)))  # function in task1
#T=1
#y = 100 + torch.cos(2*PI*(x/T)) # function in task2

plt.figure(figsize=(6,6), dpi=80)
plt.figure(1)
figs = list()
# number of epochs
n_epochs = 10000
plot_epoch = 2500
i=1    
for t in range(n_epochs+1):
    prediction = net(x)  # input x and predict based on x
    loss = loss_func(prediction, y)  # calculate loss and gradient must be (1. nn output, 2. target)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % plot_epoch == (plot_epoch-1):
        # plot and show learning process
        print(i)
        ax = "ax"+str(i)
        ax = plt.subplot(2, 2, i)
        i+=1
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5)
        plt.text(0.5,
                 0,
                 'Loss=%.4f' % loss.data.numpy(),
                 fontdict={
                     'size': 20,
                     'color': 'red'
                 })
        '''
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5,
                 0,
                 'Loss=%.4f' % loss.data.numpy(),
                 fontdict={
                     'size': 20,
                     'color': 'red'
                 })
        '''
        title = "Epoch:"+str(t)+" Hidden Neurons:"+str(n_hidden)
        plt.title(title, fontsize="small", fontweight="bold")
        #plt.pause(0.1)

#plt.ioff()
plt.show()