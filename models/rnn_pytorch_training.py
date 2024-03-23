# Import  libraries
import os
import torch
from argparse import Namespace
from torch import nn
from enums import enums_rnn_pytorch as enums
import textCorpus.brown as brown
from models.rnn_pytorch_models import RNN_v2, RNN_stack, RNNStandard
from models.utilities import l2_loss
import sys
sys.path.insert(1,'../')
print(sys.path)


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
    train_dataset, test_dataset = brown.train_test_slit(dataset)
    print("-----------------Initialization of Params------------------------")
    input_size = len(mapping)
    embedding_size = enums.EMBEDDING_SIZE
    hidden_size = enums.HIDDEN_SIZE
    output_size = input_size
    learning_rate = enums.LEARNING_RATE
    epochs = enums.EPOCHS
    mini_batch_size = enums.MINI_BATCH_SIZE
    stack_length = enums.STACK_LENGTH
    sequence_length = enums.SEQ_LENGTH
    print("---------------------------------->", sequence_length)

    print("Device : ", enums.DEVICE)

    print("----------------Creating RNN Pytorch Model-----------------------")
    if args.model == "rnn_pytorch" :
        model = RNN_v2(input_size=input_size, embedding_size=embedding_size,
                       hidden_size=hidden_size, output_size=output_size)
    elif args.model == "rnn_pytorch_stack" :
        model = RNN_stack(input_size=input_size, embedding_size=embedding_size,
                       hidden_size=hidden_size, output_size=output_size,
                          stack_length= stack_length, device = enums.DEVICE)
    elif args.model == 'rnn_pytorch_standard' :
        model = RNNStandard(input_size=input_size, embedding_size=embedding_size,
                          hidden_size=hidden_size, num_layers= stack_length,
                         device=enums.DEVICE)


    model.to(enums.DEVICE)
    print(type(model))

    if os.path.exists(enums.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(enums.CHECKPOINT_PATH))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # if args.model == "rnn_pytorch" or args.model == "rnn_pytorch_standard" :
    #     model = train_model(model=model, criterion=criterion, optimizer=optimizer, epochs=epochs,
    #                         mini_batch_size=mini_batch_size, train_dataset=train_dataset, test_dataset=test_dataset,
    #                         mapping=mapping, device=enums.DEVICE, checkpoint_path=enums.CHECKPOINT_PATH)
    # elif args.model == "rnn_pytorch_stack" :
    #     print("Yes")
    #     model = train_model_stack(model = model, criterion = criterion, optimizer = optimizer,
    #                               epochs= epochs, mini_batch_size= mini_batch_size,
    #                 train_dataset=train_dataset, test_dataset=test_dataset, mapping=mapping, device=enums.DEVICE,
    #                   checkpoint_path=enums.CHECKPOINT_PATH, stack_length=stack_length)
    #
    #     model = train_model(model=model, criterion=criterion, optimizer=optimizer, epochs=epochs,
    #                         mini_batch_size=mini_batch_size, train_dataset=train_dataset, test_dataset=test_dataset,
    #                         mapping=mapping, device=enums.DEVICE, checkpoint_path=enums.CHECKPOINT_PATH)

    model = train_model(model=model, criterion=criterion, optimizer=optimizer, epochs=epochs,
                        mini_batch_size=mini_batch_size, train_dataset=train_dataset, test_dataset=test_dataset,
                        mapping=mapping, device=enums.DEVICE, checkpoint_path=enums.CHECKPOINT_PATH)



    print("------------------------Saving Model Details----------------------------------")
    torch.save(model.state_dict(), enums.CHECKPOINT_PATH)

    print("-----------------------Evaluation Metrics-------------------------------------")
    model.load_state_dict(torch.load(enums.CHECKPOINT_PATH))



def train_model(model, criterion, optimizer, epochs, mini_batch_size,
                        train_dataset, test_dataset, mapping, device, checkpoint_path) :



    train_dataset = torch.tensor(train_dataset)
    test_dataset = torch.tensor(test_dataset)
    train_data_loader, test_data_loader = brown.transform_dataLoader(train_dataset=train_dataset,
                                                                     test_dataset=test_dataset, batch_size=mini_batch_size)

    print("----------------Training Model---------------------------")
    for epoch in range(epochs):
        for batch_idx, (data) in enumerate(train_data_loader):
            model.train()
            hidden_state = model.init_hidden()

            data = torch.tensor(data)
            data_onehot = torch.nn.functional.one_hot(data, num_classes= len(list(mapping.keys())))
            data_onehot = data_onehot.float()
            data_onehot.to(device)
            input_vector = data_onehot[:, :-1, :]
            output_vector = data_onehot[:, 1:, :]

            hidden_state = hidden_state.to(device)
            input_vector = input_vector.to(device)
            output_vector = output_vector.to(device)

            output_hat_vector = model(input = input_vector, hidden = hidden_state)
            loss = criterion(output_vector, output_hat_vector)
            loss += l2_loss(model, lambda_l2=0.01) # L2 regularization
            optimizer.zero_grad()  # setting the initial gradient to 0
            loss.backward()  # back-propagating the loss
            optimizer.step()  # updating the weights and bias values for every single step.

            print(f"Epoch : {epoch + 1}, Min-batch : {batch_idx + 1}, training-loss : {loss}")

            # Saving Checkpoints for model
            if batch_idx % enums.CHECKPOINT_FREQ == 0:
                if loss < enums.BEST_LOSS - enums.MIN_LOSS_IMPROVEMENT:
                    enums.BEST_LOSS = loss
                    torch.save(model.state_dict(), checkpoint_path) # Saving Model
                print(f"Model saved at : Epoch : {epoch + 1}, Min-batch : {batch_idx + 1}")
    return model