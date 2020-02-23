#
from torch.distributions.normal import Normal
import torch




def get_real_sampler(mu, sigma):
    dist = Normal( mu, sigma )
    return lambda m, n: dist.sample( (m, n) ).requires_grad_()

def get_noise_sampler():
    return lambda m, n: torch.rand(m, n).requires_grad_()  # Uniform-dist data into generator, _NOT_ Gaussian

#draw the graph
import matplotlib.pyplot as plt

def draw( data ) :
    plt.figure()
    d = data.tolist() if isinstance(data, torch.Tensor ) else data
    plt.plot( d )
    plt.show()