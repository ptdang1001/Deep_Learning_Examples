#paramaters

data_mean = 3.0
data_stddev = 0.2
Series_Length = 30


g_input_size = 20
g_hidden_size = 150
g_output_size = Series_Length

d_input_size = Series_Length
d_hidden_size = 75
d_output_size = 1

d_minibatch_size = 15
g_minibatch_size = 10

num_epochs = 1000
print_interval = 100

d_learning_rate = 3e-3
g_learning_rate = 8e-3
