# Import  libraries
import os
import sys
sys.path.insert(1 ,'../')
from textCorpus import brown
import torch
from torch import nn


# Import  libraries
import os
import sys
import torch
from torch import nn
import torch


import pycompss.interactive as ipycompss
if 'BINDER_SERVICE_HOST' in os.environ:
    ipycompss.start(graph=True,
                    project_xml='../xml/project.xml',
                    resources_xml='../xml/resources.xml')
else:
    ipycompss.start(graph=True, monitor=1000, debug=True, trace=True)  #


class RNN_stack(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, stack_length):
        super(RNN_stack, self).__init__()
        '''
        Input Size : 1 * |V| 
        Embedding Size : d 
        Embedding Matrix : |V| * d 
        '''
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.stack_length = stack_length

        self.embedding = nn.Linear(in_features=input_size, out_features=embedding_size, bias=False)
        self.weights = []
        self.u_ls = []
        self.v_ls = []
        for i in range(stack_length):
            self.weight = nn.Linear(in_features=embedding_size, out_features=hidden_size, bias=False).to(device)
            self.weights.append(self.weight)
            self.u = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False).to(device)
            self.u_ls.append(self.u)
            self.v = nn.Linear(in_features=hidden_size, out_features=embedding_size, bias=False).to(device)
            self.v_ls.append(self.v)

    def encode(self, input):
        e = self.embedding(input)
        return e

    def softmax(self, input):
        out = torch.softmax(input, dim=1)
        return out

    def rnn_cell(self, input, hidden, stack_idx):
        input = input.to(device)
        hidden = hidden.to(device)
        h1 = self.weights[stack_idx](input).to(device)
        h2 = self.u_ls[stack_idx](hidden)
        h = torch.add(h1, h2)
        out = self.v_ls[stack_idx](h)
        return out, h

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(self.stack_length, self.hidden_size))

    def forward(self, input, hidden_states, stack_length):
        # print("Input : ", input.shape)
        # print("Hidden State : ", hidden_states.shape)
        # print("Stack Len : ", stack_length)
        input = self.embedding(input)
        for stack_idx in range(stack_length):
            output_hat_ls = []
            hidden_state = hidden_states[stack_idx]
            hidden_state = hidden_state.to(device)
            for sequence_length_idx in range(input.shape[1]):
                output_hat_vector, hidden_state = self.rnn_cell(input[:, sequence_length_idx, :], hidden_state,
                                                                stack_idx)
                output_hat_ls.append(output_hat_vector)

            output_hat_stack = torch.stack(output_hat_ls, dim=1)
            output_hat_stack = output_hat_stack.to(device)
            input = output_hat_stack
            input = input.to(device)
        output_hat_stack = nn.Linear(in_features=embedding_size, out_features=self.output_size, bias=False).to(device)(
            output_hat_stack)
        output_hat_stack = self.softmax(output_hat_stack)
        return output_hat_stack


def get_rnn_stack_parameter() -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    checkpoint_dir = "../checkpoints/rnn_pytorch/"
    checkpoint_path = os.path.join(checkpoint_dir, f"rnn_pytorch_stack.pth")
    model = torch.load(checkpoint_path)

    embedding_paramaters = []
    for param in model.embedding.parameters():
        embedding_paramaters.append(param)

    embedding_paramaters = embedding_paramaters[0]

    weight_paramaters = []
    for i in model.weights:
        for param in i.parameters():
            weight_paramaters.append(param)

    weight_paramaters = torch.stack(weight_paramaters, dim= 0)

    u_paramaters = []
    for i in model.u_ls:
        for param in i.parameters():
            u_paramaters.append(param)

    u_paramaters = torch.stack(u_paramaters, dim= 0)

    v_paramaters = []
    for i in model.v_ls:
        for param in i.parameters():
            v_paramaters.append(param)

    v_paramaters = torch.stack(v_paramaters, dim=0)

    return embedding_paramaters, weight_paramaters, u_paramaters, v_paramaters

def init_hidden(stack_length, hidden_size, mini_batch_size):
    hidden = []
    for i in range(stack_length) :
        mini_batch_hidden = []
        value = nn.init.kaiming_uniform_(torch.empty(1, hidden_size))
        for i in range(mini_batch_size) :
            mini_batch_hidden.append(value)
        mini_batch_hidden = torch.concat(mini_batch_hidden)
        hidden.append(mini_batch_hidden)

    hidden = torch.stack(hidden, dim=0)
    return hidden


def embedding_convert(input_vector, embedding):
    output = torch.matmul(input_vector, torch.t(embedding))
    return output


def rnn_cell_computation(input_vector, hidden_state, weight, u, v, embedding):
    h1 = torch.matmul(input_vector, torch.t(weight))
    h2 = torch.matmul(hidden_state, torch.t(u))
    h = torch.add(h1, h2)
    out = torch.matmul(h, torch.t(v))
    return out, h

def output_one_hot(output, embedding) :
    out = torch.matmul(output, embedding)
    out = torch.softmax(out,  dim=1)
    return out

stack_length = 4
sequence_length = 4
device = "cpu"
hidden_size = 256
mini_batch_size = 512
e,w,u,v = get_rnn_stack_parameter()
e = e.to(device)
w = w.to(device)
u = u.to(device)
v = v.to(device)
h = init_hidden(stack_length = 4, hidden_size= hidden_size, mini_batch_size= mini_batch_size)
print("Embedding : ", e.shape)
print("Weight : ", w.shape)
print("U : ", u.shape)
print("V : ", v.shape)
print("Hidden : ", h.shape)

print("-----------------Loading Dataset---------------------------------")
dataset, mapping, reverse_mapping = brown.dataset()
train_dataset, test_dataset = brown.train_test_slit(dataset)

train_dataset = torch.tensor(train_dataset)
test_dataset = torch.tensor(test_dataset)
train_data_loader, test_data_loader = brown.transform_dataLoader(train_dataset=train_dataset,
                                                                     test_dataset=test_dataset, batch_size=mini_batch_size)
for batch_idx, (data) in enumerate(train_data_loader):
    data = torch.tensor(data)
    data_onehot = torch.nn.functional.one_hot(data, num_classes=len(list(mapping.keys())))
    data_onehot = data_onehot.float()
    input_vector = data_onehot[:, :-1, :]
    print("Input Vector : ", input_vector.shape)
    print("Hidden Vector : ", h.shape)
    input_vector = embedding_convert(input_vector = input_vector, embedding= e)
    output_vector, hidden_state = rnn_cell_computation(input_vector= input_vector[:, 0 , :], hidden_state= h[0, :, :], weight= w[0, :, :], u= u[0, :, :] , v= v[0, :, :], embedding= e)

    output_vector = [[None for i in range(sequence_length)] for j in range(stack_length)]
    hidden_vector = [[None for i in range(sequence_length)] for j in range(stack_length)]

    # First Stack Layer
    hidden_state = h[0, :, :]
    for i in range(sequence_length) :
        output_vector[0][i], hidden_state = rnn_cell_computation(input_vector = input_vector[:, i, :], hidden_state= hidden_state, weight= w[0, :, :], u = u[0, :, :],v = v[0, :, :], embedding= e)

    for i in range(1, stack_length) :
        hidden_state = h[i, :, :]
        for j in range(sequence_length) :
            output_vector[i][j], hidden_state = rnn_cell_computation(input_vector = output_vector[i-1][j], hidden_state= hidden_state, weight= w[i, :, :], u = u[i, :, :],v = v[i, :, :], embedding= e)

    for i in range(sequence_length) :
        output_vector[-1][i] = output_one_hot(output_vector[-1][i], embedding= e)

    if batch_idx == 5 :
        break
print("Done")



