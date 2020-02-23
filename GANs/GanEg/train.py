import torch
import torch.nn as nn
import torch.optim as optim

#my packages
from parameters import *
import modules
import tools




real_data = tools.get_real_sampler( data_mean, data_stddev )
noise_data  = tools.get_noise_sampler()

G = modules.Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = modules.Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)


criterion = nn.BCELoss() #Binary Cross Entropy
#optimizers of d and g
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate ) #, betas=optim_betas)
g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate ) #, betas=optim_betas)

def train_D_on_real() :
    real_data = real_data( d_minibatch_size, d_input_size )
    decision = D( real_data )
    error = criterion( decision, torch.ones( d_minibatch_size, 1 ))  # ones = true
    error.backward()

def train_D_on_fake() :
    noise = noise_data( d_minibatch_size, g_input_size )
    fake_data = G( noise )
    decision = D( fake_data )
    error = criterion( decision, torch.zeros( d_minibatch_size, 1 ))  # zeros = fake
    error.backward()

def train_G_on_fake():
    noise = noise_data( g_minibatch_size, g_input_size )
    fake_data = G( noise )
    fake_decision = D( fake_data )
    error = criterion( fake_decision, torch.ones( g_minibatch_size, 1 ) )  # we want to fool, so pretend it's all genuine

    error.backward()
    return error.item(), fake_data


losses = []

for epoch in range(num_epochs):
    D.zero_grad()#clear the parameters

    #train D
    train_D_on_real()
    train_D_on_fake()
    d_optimizer.step()

    #train G
    G.zero_grad()
    loss, generated = train_G()
    g_optimizer.step()

    #loss
    losses.append(loss)

    #print the training process
    if (epoch % print_interval) == (print_interval - 1):
        print("Epoch %6d. Loss %5.3f" % (epoch + 1, loss))

print("Training complete")




d = torch.empty( generated.size(0), 53 )
for i in range( 0, d.size(0) ) :
    d[i] = torch.histc( generated[i], min=0, max=5, bins=53 )
tools.draw( d.t() )