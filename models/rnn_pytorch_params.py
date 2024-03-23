# Import  libraries
import os
import torch
import textCorpus.brown as brown
from enums import  enums_rnn_pytorch as enums
from argparse import Namespace

from models.rnn_pytorch_models import RNN_v2, RNN_stack


def main(args : Namespace) :
    '''
            Main function to train and generate predictions in csv format

            Args:
            - args : Namespace : command line arguments
    '''

    enums.EPOCHS = args.num_iters
    enums.MINI_BATCH_SIZE = args.batch_size
    enums.CHECKPOINT_PATH = args.checkpoint_path
    enums.LEARNING_RATE = args.lr
    enums.L2_LAMBDA = args.l2_lambda
    enums.DEVICE = args.device
    enums.STACK_LENGTH = args.stack_length
    enums.SEQ_LENGTH = args.sequence_length

    print("-----------------Loading Dataset---------------------------------")
    dataset, mapping, reverse_mapping = brown.dataset()
    print("-----------------Initialization of Params------------------------")
    input_size = len(mapping)
    embedding_size = enums.EMBEDDING_SIZE
    hidden_size = enums.HIDDEN_SIZE
    output_size = input_size
    print("Device : ", enums.DEVICE)

    print("----------------Creating RNN Pytorch Model-----------------------")

    if args.model == "rnn_pytorch" :
        model = RNN_v2(input_size=input_size, embedding_size=embedding_size,
                       hidden_size=hidden_size, output_size=output_size)
    elif args.model == "rnn_pytorch_stack" :
        model = RNN_stack(input_size=input_size, embedding_size=embedding_size,
                       hidden_size=hidden_size, output_size=output_size,
                          stack_length= enums.STACK_LENGTH, device = enums.DEVICE)

    if args.model == "rnn_pytorch" :
        model.load_state_dict(torch.load(enums.CHECKPOINT_PATH))
        model.to(enums.DEVICE)
        print(type(model))

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

    if args.model == "rnn_pytorch_stack":
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

    print("Embedding", len(embedding_paramaters))
    print("U Params ", len(u_paramaters))
    print("V Params", len(v_paramaters))
    print("W Params", len(weight_paramaters))

    return embedding_paramaters, weight_paramaters, u_paramaters, v_paramaters
