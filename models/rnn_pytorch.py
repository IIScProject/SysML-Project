# Import  libraries
import os
import sys
import torch
from torch import nn
import textCorpus.brown as brown
from torchsummary import  summary
import utilities as utilities
sys.path.insert(1,'../')


'''
Input  : Sentence (List of tokens for a sentence)
         mapping : Key : Word  Value : Index
         reverse_mapping : Key : Index , Value : Word

Output : Return the nested list of token in one hot vector form 
'''

# class RNN(nn.Module) :
#     def __init__(self, input_size , embedding_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         '''
#         Input Size : 1 * |V|
#         Embedding Size : d
#         Embedding Matrix : |V| * d
#
#
#         '''
#         self.input_size = input_size
#         self.embedding_size = embedding_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#
#         self.embedding = nn.Linear(in_features= input_size, out_features= embedding_size, bias= False )
#         self.weight = nn.Linear(in_features= embedding_size, out_features= hidden_size, bias= False)
#         self.u = nn.Linear(in_features= hidden_size, out_features= hidden_size, bias= False)
#         self.v = nn.Linear(in_features= hidden_size, out_features= input_size, bias= False)
#
#     def forward(self, input, hidden_state):
#         e = self.embedding(input)
#         h1 = self.weight(e)
#         h2 = self.u(hidden_state)
#         h = torch.add(h1, h2)
#         out = self.v(h)
#         out = torch.softmax(out, dim = 1 )
#         out.reshape(-1)
#         return out, h
#
#     def init_hidden(self):
#         return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))


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


def train_model(model, criterion, optimizer, epochs, mini_batch_size, train_dataset, test_dataset) :
    print("----------------Training Model---------------------------")
    for epoch in range(epochs):
        model.train()
        mini_batches = len(train_dataset) // mini_batch_size
        idx = 0
        for mini_batch in range(mini_batches):
            mini_batch_loss = 0
            hidden_state = model.init_hidden()
            total_dividend = 0
            for _ in range(mini_batch_size):
                sentence = train_dataset[idx]
                idx += 1
                idx = idx % len(train_dataset)
                # Converting two one-hot vectors
                sentence = brown.convert_sentence_to_one_hot_tensor(sentence=sentence, mapping=mapping)
                sentence_loss = 0
                total_dividend += len(sentence) - 1
                for sentence_idx in range(len(sentence) - 1):
                    input_vector = sentence[sentence_idx].float().reshape(1, -1)
                    output_vector = sentence[sentence_idx + 1].float().reshape(1, -1)
                    output_hat, hidden_state = model(input_vector, hidden_state)
                    loss = criterion(output_vector, output_hat)
                    if sentence_loss == 0:
                        sentence_loss = loss
                    else:
                        sentence_loss += loss
                if mini_batch_loss == 0:
                    mini_batch_loss = sentence_loss
                else:
                    mini_batch_loss += sentence_loss
            optimizer.zero_grad()
            mini_batch_loss /= total_dividend
            mini_batch_loss += utilities.l2_loss(model, lambda_l2=0.01)
            mini_batch_loss.backward()
            optimizer.step()
            print(f"Epoch : {epoch + 1}, Min-batch : {mini_batch + 1}, training-loss : {mini_batch_loss}")

    return model


# def train_model_cpu_gpu(model, criterion, optimizer, epochs, mini_batch_size, train_dataset, test_dataset, mapping, device) :
#     train_dataset = torch.tensor(train_dataset)
#     test_dataset = torch.tensor(test_dataset)
#     train_data_loader, test_data_loader = brown.transform_dataLoader(train_dataset=train_dataset,
#                                                                      test_dataset=test_dataset, batch_size=mini_batch_size)
#
#
#     print("----------------Training Model---------------------------")
#     for epoch in range(epochs):
#         for batch_idx, (data) in enumerate(train_data_loader):
#             model.train()
#             hidden_state = model.init_hidden()
#
#             data = torch.tensor(data)
#             data_onehot = torch.nn.functional.one_hot(data, num_classes= len(list(mapping.keys())))
#             data_onehot = data_onehot.float()
#             data_onehot.to(device)
#             input_vector = data_onehot[:, :-1, :]
#             output_vector = data_onehot[:, 1:, :]
#
#             hidden_state = hidden_state.to(device)
#             input_vector = input_vector.to(device)
#             output_vector = output_vector.to(device)
#
#             loss = 0
#             for sequence_length_idx in range(input_vector.shape[1]):
#                 output_hat_vector, hidden_state = model(input_vector[:, sequence_length_idx, :], hidden_state)
#                 if loss == 0:
#                     loss = criterion(output_vector[:, sequence_length_idx, :], output_hat_vector)
#                 else:
#                     loss += criterion(output_vector[:, sequence_length_idx, :], output_hat_vector)
#
#             loss += utilities.l2_loss(model, lambda_l2=0.01)
#             optimizer.zero_grad()  # setting the initial gradient to 0
#             loss.backward()  # back-propagating the loss
#             optimizer.step()  # updating the weights and bias values for every single step.
#
#             print(f"Epoch : {epoch + 1}, Min-batch : {batch_idx + 1}, training-loss : {loss}")
#
#     return model
#

def train_model_cpu_gpu(model, criterion, optimizer, epochs, mini_batch_size, train_dataset, test_dataset, mapping, device) :

    # Define checkpointing parameters
    checkpoint_dir = "../checkpoints/rnn_pytorch/"
    checkpoint_frequency = 10  # Save checkpoint after every 100 minibatches
    min_loss_improvement = 0.0001  # Minimum improvement in loss to save a new checkpoint
    best_loss = float('inf')


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
            loss += utilities.l2_loss(model, lambda_l2=0.01) # L2 regularization
            optimizer.zero_grad()  # setting the initial gradient to 0
            loss.backward()  # back-propagating the loss
            optimizer.step()  # updating the weights and bias values for every single step.

            print(f"Epoch : {epoch + 1}, Min-batch : {batch_idx + 1}, training-loss : {loss}")

            # Saving Checkpoints for model
            if batch_idx % checkpoint_frequency == 0:
                if loss < best_loss - min_loss_improvement:
                    best_loss = loss
                    # Saving Model
                    checkpoint_path = os.path.join(checkpoint_dir, f"rnn_pytorch.pth")
                    torch.save(model, checkpoint_path)
                print(f"Model saved at : Epoch : {epoch + 1}, Min-batch : {batch_idx + 1}")
    return model


dataset, mapping, reverse_mapping = brown.dataset()
train_dataset, test_dataset = brown.train_test_slit(dataset)

input_size = len(mapping)
embedding_size = 300
hidden_size = 256
output_size = input_size
learning_rate = 0.01
epochs = 100
mini_batch_size = 1024

model = RNN_v2(input_size = input_size, embedding_size= embedding_size,
            hidden_size= hidden_size, output_size= output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("-------------------------Model Summary----------------------------------------")
print(summary(model, input_size = [(1, input_size), (1, hidden_size)] , batch_size= -1))

model = train_model_cpu_gpu(model= model, criterion= criterion, optimizer= optimizer, epochs= epochs,
                            mini_batch_size= mini_batch_size, train_dataset= train_dataset, test_dataset= test_dataset,
                            mapping= mapping, device= device)

print("------------------------Saving Model Details----------------------------------")
torch.save(model, "rnn_pytorch.pth")

print("-----------------------Evaluation Metrics-------------------------------------")
model = torch.load("rnn_pytorch.pth")
