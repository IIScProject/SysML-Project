# Import  libraries
import os
import sys
sys.path.insert(1,'../')
import torch
from torch import nn
import textCorpus.brown as brown
import utilities as utilities
from enums import  enums_rnn_pytorch as enums
import rnn_pytorch_models as models

def get_params() :

    print("-----------------Loading Dataset---------------------------------")
    dataset, mapping, reverse_mapping = brown.dataset()
    train_dataset, test_dataset = brown.train_test_slit(dataset)
    print("-----------------Initialization of Params------------------------")
    input_size = len(mapping)
    embedding_size = enums.EMBEDDING_SIZE
    hidden_size = enums.HIDDEN_SIZE
    output_size = input_size
    learning_rate = enums.LEARNING_RATE
    epochs = enums.EPOCHS
    mini_batch_size = enums.MINI_BATCH_SIZE
    print("Device : ", enums.DEVICE)

    print("----------------Creating RNN Pytorch Model-----------------------")
    model = models.RNN_v2(input_size = input_size, embedding_size= embedding_size,
                hidden_size= hidden_size, output_size= output_size)
    model.to(enums.DEVICE)
    print(type(model))

    # Loading model from checkpoint if exists
    checkpoint_dir = enums.CHECKPOINT_DIR
    checkpoint_path = os.path.join(checkpoint_dir, enums.CHECKPOINT_FILE_RNN_V2)
    if enums.CHECKPOINT_PATH is None :
        enums.CHECKPOINT_PATH = checkpoint_path

    model.load_state_dict(torch.load(enums.CHECKPOINT_PATH))

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

    # v parameters
    v_params = model.v.parameters()
    v_paramaters = []
    for param in v_params:
        v_paramaters.append(param.clone().detach())


    return embedding_paramaters[0],weight_paramaters[0],u_paramaters[0],v_paramaters[0]
