import os
#os.chdir('C:/Users/wnchang/Documents/F/PhD_Course/ECE629_NN/4.Project/Proj1')
import sys
import numpy as np
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data

PI = 3.14
   
   
#    # this is a Network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer
    
    def forward(self, x):
        x = F.sigmoid(self.hidden(x))  # ReLu function for hidden layer
        x = self.predict(x)  # linear output
        return x

def main():

    n_feature = 1  # the number of input neurons
    n_hidden = 50  # the number of hidden neurons
    n_output = 1  # the number of output neurons
    net = Net(n_feature, n_hidden, n_output)  # define the network
    print(net)  # net architecture
    
    
    x= torch.unsqueeze(torch.linspace(0, 24, 1000),dim=1)  # x data (tensor), shape=(100, 1)
    y = torch.sin(PI * x/12)  + 100

    
    x, y = Variable(x), Variable(y)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    
    #    BATCH_SIZE = 100
    EPOCH = 6000
    PLOT_EPOCH = 1500
    
    plt.figure(figsize=(6,6), dpi=80)
    plt.figure(1)

    lossValue = []
    # start training
    start = time.time()
    i=1
    for epoch in range(1, EPOCH+1):
        
        if epoch % 100 == 0:
                print("epoch: ", epoch )
        
#        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            
#        b_x = Variable(batch_x)
#        b_y = Variable(batch_y)
        b_x = x
        b_y = y

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
        lossValue.append(loss)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        
        if epoch % PLOT_EPOCH == 0:

            print(i)
            ax = "ax"+str(i)
            ax = plt.subplot(2, 2, i)
            i+=1
            plt.scatter(x.data.numpy(), y.data.numpy(), alpha=0.5)
            plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5, alpha=0.5)
            plt.text(0.5,
                 0,
                 'Loss=%.4f' % loss.data.numpy(),
                 fontdict={
                     'size': 20,
                     'color': 'red'
                 })
        
            title = "Epoch:"+str(epoch)+"lr:"+str(0)+" Hidden Neurons:"+str(0)
            plt.title(title, fontsize="small", fontweight="bold")
    end = time.time()
    print("Training time: ", end - start) 
    plt.ioff()
    plt.show()

#    timestr = time.strftime("%Y%m%d-%H%M%S")
#    imgName = "./Task1_" + timestr + ".png"
#    plt.savefig(imgName)
    '''
    plt.figure(figsize=(7,7))
    imgLoss = "./Task_1_loss" + timestr + ".png"
    plt.plot(lossValue)
    plt.title("MSE Loss Value")
    plt.savefig(imgLoss)
    '''
    return 

if __name__ == "__main__":
    main()