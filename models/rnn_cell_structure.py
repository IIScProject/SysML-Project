import torch
from torch import nn
from torch.utils.data import DataLoader

import enums.enums_rnn_pytorch
from textCorpus import brown
import time


def rnn_cell_computation(input_vector : torch.Tensor, hidden_state : torch.Tensor, weight : torch.Tensor, u : torch.Tensor, v : torch.Tensor, embedding : torch.Tensor):
    h1 = compute_1(input_vector, weight)
    h2 = compute_2(hidden_state, u)
    h = compute_3(h1, h2)
    out = compute_4(hidden_state, v)

    return out, h

def compute_1(input_vector : torch.Tensor, weight : torch.Tensor) -> torch.Tensor:
    return torch.matmul(input_vector, torch.t(weight))

def compute_2(hidden_state : torch.Tensor,  u : torch.Tensor) -> torch.Tensor :
    return torch.matmul(hidden_state, torch.t(u))

def compute_3(hidden_state_1 : torch.Tensor, hidden_state_2 : torch.Tensor) -> torch.Tensor :
    return torch.add(hidden_state_1, hidden_state_2)

def compute_4(hidden_state : torch.Tensor, v : torch.Tensor) -> torch.Tensor :
    return torch.matmul(hidden_state, torch.t(v))

def embedding_convert(input_vector, embedding):
    output = torch.matmul(input_vector, torch.t(embedding))
    return output

def output_one_hot(output, embedding) :
    out = torch.matmul(output, embedding)
    out = torch.softmax(out,  dim=1)
    return out

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



def inference(dataset, mapping, reverse_mapping, embedding_paramaters, weight_paramaters, u_paramaters, v_paramaters, batch_size, sequence_length, stack_length, hidden_size) :

    # Start Time
    start_time = time.time()
    train_dataset, test_dataset = brown.train_test_slit(dataset)
    train_dataset = torch.tensor(train_dataset)
    test_dataset = torch.tensor(test_dataset)
    train_data_loader, test_data_loader = brown.transform_dataLoader(train_dataset= train_dataset, test_dataset= test_dataset, batch_size= batch_size)
    embedding_paramaters = embedding_paramaters[0]

    for batch_idx, (data) in enumerate(test_data_loader) :

        data = torch.tensor(data)
        data_onehot = torch.nn.functional.one_hot(data, num_classes=len(list(mapping.keys())))
        data_onehot = data_onehot.float()
        data_onehot.to(enums.enums_rnn_pytorch.DEVICE)
        input_vector = data_onehot[:, :-1, :]
        h = init_hidden(stack_length=4, hidden_size=hidden_size, mini_batch_size=batch_size)
        # print("Input Vector : ", input_vector.shape)
        # print("Hidden Vector : ", h.shape)
        # print("Embedding Params :", embedding_paramaters.shape)
        # print("Weight Params : ", type(weight_paramaters), "Length : ", len(weight_paramaters), "Shape : ", weight_paramaters[0].shape )
        # print("U Params : ", type(u_paramaters), "Length :  ", len(u_paramaters), "Shape : ", u_paramaters[0].shape)
        # print("V Prams :", type(v_paramaters), "Length : ", len(v_paramaters), "Shape : ", v_paramaters[0].shape)
        input_vector = embedding_convert(input_vector=input_vector, embedding=embedding_paramaters)

        output_vector = [[None for i in range(sequence_length-1)] for j in range(stack_length)]
        hidden_state = [h[j, :, :] for j in range(stack_length)]

        # Combining all layers
        for i in range(stack_length) :
            for j in range(sequence_length-1) :
                if i == 0 :
                    output_vector[i][j], hidden_state[i] = rnn_cell_computation(input_vector=input_vector[:, j, :],
                                                                             hidden_state=hidden_state[i],
                                                                             weight=weight_paramaters[i],
                                                                             u=u_paramaters[i], v=v_paramaters[i],
                                                                             embedding=embedding_paramaters)
                else :
                    output_vector[i][j], hidden_state[i] = rnn_cell_computation(input_vector=output_vector[i - 1][j],
                                                                             hidden_state=hidden_state[i],
                                                                             weight=weight_paramaters[i],
                                                                             u=u_paramaters[i], v=v_paramaters[i],
                                                                             embedding=embedding_paramaters)

        # Converting back to one hot
        for i in range(sequence_length-1):
            output_vector[-1][i] = output_one_hot(output_vector[-1][i], embedding=embedding_paramaters)

    # End Time
    end_time = time.time()
    return


