# Import  libraries
import os
import sys
import torch
from torch import nn

class RNN_v2(nn.Module):
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

    def rnn_cell(self, input, hidden):
        e = self.embedding(input)
        h1 = self.weight(e)
        h2 = self.u(hidden)
        h = torch.add(h1, h2)
        out = self.v(h)
        out = torch.softmax(out, dim=1)
        out.reshape(-1)
        return out, h

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))

    def forward(self, input, hidden):
        output_hat_ls = []
        for sequence_length_idx in range(input.shape[1]):
            output_hat_vector, hidden = self.rnn_cell(input[:, sequence_length_idx, :], hidden)
            output_hat_ls.append(output_hat_vector)

        output_hat_stack = torch.stack(output_hat_ls, dim=1)
        return output_hat_stack


class RNN_stack(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, stack_length, device):
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
        self.device = device

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
        input = input.to(self.device)
        hidden = hidden.to(self.device)
        h1 = self.weights[stack_idx](input).to(self.device)
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
            hidden_state = hidden_state.to(self.device)
            for sequence_length_idx in range(input.shape[1]):
                output_hat_vector, hidden_state = self.rnn_cell(input[:, sequence_length_idx, :], hidden_state,
                                                                stack_idx)
                output_hat_ls.append(output_hat_vector)

            output_hat_stack = torch.stack(output_hat_ls, dim=1)
            output_hat_stack = output_hat_stack.to(self.device)
            input = output_hat_stack
            input = input.to(self.device)
        output_hat_stack = nn.Linear(in_features=self.embedding_size, out_features=self.output_size, bias=False).to(self.device)(
            output_hat_stack)
        output_hat_stack = self.softmax(output_hat_stack)
        return output_hat_stack

