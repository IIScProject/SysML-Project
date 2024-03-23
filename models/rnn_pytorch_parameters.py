import os
import sys

from textCorpus import brown

sys.path.insert(1,'../')
import torch
import torchvision.models as models
from torch import nn


class RNN_v2(torch.nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(RNN_v2, self).__init__()
        '''
        Input Size : 1 * |V| 
        Embedding Size : d 
        Embedding Matrix : |V| * d 
        '''
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Linear(in_features=input_size, out_features=embedding_size, bias=False)
        self.weight = nn.Linear(in_features=embedding_size, out_features=hidden_size, bias=False)
        self.u = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.v = nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)

# torch_model = torch.load("/Users/avneeshgautam29/Documents/SysML/SysML-Project/checkpoints/rnn_pytorch/rnn_pytorch.pth")

def get_rnn_parameter():
    print("-----------------Loading Dataset---------------------------------")
    dataset, mapping, reverse_mapping = brown.dataset()
    train_dataset, test_dataset = brown.train_test_slit(dataset)
    print("-----------------Initialization of Params------------------------")
    input_size = len(mapping)
    embedding_size = 300
    hidden_size = 256
    output_size = input_size
    learning_rate = 0.01
    epochs = 100
    mini_batch_size = 512

    print("----------------Creating RNN Pytorch Model-----------------------")
    model = RNN_v2(input_size=input_size, embedding_size=embedding_size,
                   hidden_size=hidden_size, output_size=output_size)

    checkpoint_dir = "../checkpoints/rnn_pytorch/"
    checkpoint_path = os.path.join(checkpoint_dir, f"rnn_pytorch.pth")
    # model = RNN_v2()
    model.load_state_dict(torch.load(checkpoint_path))
    # embedding parameters
    embedding_params = model.embedding.parameters()
    embedding_paramaters = []
    for param in embedding_params:
        embedding_paramaters.append(param.clone().detach())

    # weight parameters
    weight_params = model.weight.parameters()
    weight_paramaters = []
    for param in weight_params:
        weight_paramaters.append(param.clone().detach())

    # u parameters
    u_params = model.u.parameters()
    u_paramaters = []
    for param in u_params:
        u_paramaters.append(param.clone().detach())

    # u parameters
    v_params = model.v.parameters()
    v_paramaters = []
    for param in v_params:
        v_paramaters.append(param.clone().detach())


    return embedding_paramaters[0],weight_paramaters[0],u_paramaters[0],v_paramaters[0]

get_rnn_parameter()

# e,w,u,v = get_rnn_parameter()
# print(e)
# print(e.weight)
# print(w)
# print(u)
# print(v)