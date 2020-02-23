
import torch
import torch.nn as nn

#generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.xfer = torch.nn.SELU()

    def forward(self, x):
        x = self.xfer( self.map1(x) ) #activation function
        x = self.xfer( self.map2(x) )
        return self.xfer( self.map3( x ) )  #return fake data

#discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        return torch.sigmoid(self.map3(x)) #return probability

'''
You can define the function like this if you have several hidden layers:
    class Discriminator(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden1_size)
        self.map2 = nn.Linear(hidden1_size, hidden2_size)
        self.map3 = nn.Linear(hidden2_size, output_size)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        return torch.sigmoid(self.map3(x))

'''