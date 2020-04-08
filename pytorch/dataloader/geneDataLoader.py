# -*- coding: utf-8 -*

'''
This is a simple example to help you know
how to load matrix, like gene data, which is not image data.
if you wanna run the code bellow, you must install pytorch and python.
Here is pytorch-1.4.0 and python-3.7.7 
'''

# load packages
import torch
import torch.utils.data as Data
from os import cpu_count
import  sys

# prepare your gene data and label
'''
this is your gene data, 
which is a tensor with 5 slides(5 scRNA-seq datasets), 
each slide has 3 rows(genes) and 3 cols(cells)
'''
geneData = torch.randn([5, 3, 3])

# this is your labels for each slide.
geneData_label = torch.linspace(1, 5, 5)


BATCH_SIZE = 2  # batch size means you will train how many slides at the same time

torch_dataset = Data.TensorDataset(geneData, geneData_label)  # To transform your data and label which can be identified by torch
#if you wanna know what the data looks like you can print them
#print(geneData)
#print(geneData_label)
#sys.exit(1) #pause here


loader = Data.DataLoader(
    dataset=torch_dataset,  # your data and label
    batch_size=BATCH_SIZE,  # batch size
    shuffle=True,  # shuffle data True or False
    #num_workers=cpu_count()  
    num_workers=2 # threads = batche groups you wanna train at the same time, you can use the cpu count
)

'''
You can put your training model below
'''
if __name__ == "__main__":  # if you use windows, please put your training process under "if __name__ == "__main__""
    print("---------------------------Training Start---------------------------------")
    for epoch in range(3): #train 3 times
        '''
        Here is your train process
        '''
        #print the data and their label to help you understand what is in the data loader.
        for step, (batchGeneData, batchGeneDataLabel) in enumerate(loader):
            print('Epoch: ', epoch, '| Step: ', step, '| batch geneData: ',
                  batchGeneData.numpy(), '| batch geneData: ', batchGeneDataLabel.numpy())
    print("----------------------------Trainning End------------------------------------")
