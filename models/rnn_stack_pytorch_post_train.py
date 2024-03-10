import os

import torch
from torch import nn

from textCorpus import brown

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
mini_batch_size = 1024
stack_length = 4

class RNN_stack(nn.Module) :

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
        for i in range(stack_length) :
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
        for stack_idx in range(stack_length) :
            output_hat_ls = []
            hidden_state = hidden_states[stack_idx]
            hidden_state = hidden_state.to(device)
            for sequence_length_idx in range(input.shape[1]):
                output_hat_vector, hidden_state = self.rnn_cell(input[:, sequence_length_idx, :], hidden_state, stack_idx)
                output_hat_ls.append(output_hat_vector)

            output_hat_stack = torch.stack(output_hat_ls, dim=1)
            output_hat_stack = output_hat_stack.to(device)
            input = output_hat_stack
            input = input.to(device)
        output_hat_stack = nn.Linear(in_features= embedding_size, out_features= self.output_size, bias= False).to(device)(output_hat_stack)
        output_hat_stack = self.softmax(output_hat_stack)
        return output_hat_stack


print("--------------------Reload the Saved Model-------------------------------")
# Loading model from checkpoint if exists
checkpoint_dir = "../checkpoints/rnn_pytorch/"
checkpoint_path = os.path.join(checkpoint_dir, f"rnn_pytorch_stack.pth")
if os.path.exists(checkpoint_path) :
    model = torch.load(checkpoint_path)

print("Checkpoint path : ", checkpoint_path)
model = torch.load(checkpoint_path)
device = torch.device('cpu')
model = model.to(device)

print(list(model.parameters()))
print("-----------------------------Loading Weights for Model-------------------------")
# Get weights for each layer
layer_weights = {}
for name, param in model.named_parameters():
    if 'weight' in name:  # Only consider weight parameters
        layer_weights[name] = param
