# Import  libraries
import os
import sys
import torch
from torch import nn
sys.path.insert(1,'../')


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

def get_rnn_stack_parameter():
    checkpoint_dir = "../checkpoints/rnn_pytorch/"
    checkpoint_path = os.path.join(checkpoint_dir, f"rnn_pytorch_stack.pth")
    print(checkpoint_path)
    model = torch.load(checkpoint_path)
    
    embedding_paramaters = []
    for param in model.embedding.parameters():
       embedding_paramaters.append(param)

    # print(embedding_paramaters)

    weight_paramaters = []
    for i in model.weights:
        for param in i.parameters():
            weight_paramaters.append(param)
    
    u_paramaters = []
    for i in model.u_ls:
        for param in i.parameters():
            u_paramaters.append(param)

    v_paramaters = []
    for i in model.v_ls:
        for param in i.parameters():
            v_paramaters.append(param)
    
    return embedding_paramaters,weight_paramaters,u_paramaters,v_paramaters

get_rnn_stack_parameter()


# e,w,u,v = get_rnn_stack_parameter()

# print("\n\n ************************* Embedding Parameters ************************************ \n")
# print("\n",e)
# print("\n\n ************************* Weight Parameters *********************************** \n")
# print("\n",w)
# print("\n\n ************************* U Parameters **************************************** \n")
# print("\n",u)
# print("\n\n ************************* V Parameters **************************************** \n")
# print("\n",v)
