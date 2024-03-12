# Import  libraries
import os
import sys
sys.path.insert(1,'../')
import torch
from torch import nn
import textCorpus.brown as brown
from torchsummary import  summary
import utilities as utilities
torch.cuda.empty_cache()


'''
Input  : Sentence (List of tokens for a sentence)
         mapping : Key : Word  Value : Index
         reverse_mapping : Key : Index , Value : Word

Output : Return the nested list of token in one hot vector form 
'''

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


def train_model_cpu_gpu(model, criterion, optimizer, epochs, mini_batch_size,
                        train_dataset, test_dataset, mapping, device, checkpoint_path) :

    # Define checkpointing parameters
    checkpoint_frequency = 50  # Save checkpoint after every 100 minibatches
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
                    torch.save(model, checkpoint_path) # Saving Model
                print(f"Model saved at : Epoch : {epoch + 1}, Min-batch : {batch_idx + 1}")
    return model


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
model = RNN_v2(input_size = input_size, embedding_size= embedding_size,
            hidden_size= hidden_size, output_size= output_size)
# Loading model from checkpoint if exists
checkpoint_dir = "../checkpoints/rnn_pytorch/"
checkpoint_path = os.path.join(checkpoint_dir, f"rnn_pytorch.pth")
if os.path.exists(checkpoint_path) :
    model = torch.load(checkpoint_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("-------------------------Model Summary----------------------------------------")
print(summary(model, input_size = [(1, input_size), (1, hidden_size)] , batch_size= -1))
model = train_model_cpu_gpu(model= model, criterion= criterion, optimizer= optimizer, epochs= epochs,
                            mini_batch_size= mini_batch_size, train_dataset= train_dataset, test_dataset= test_dataset,
                            mapping= mapping, device= device, checkpoint_path= checkpoint_path)

print("------------------------Saving Model Details----------------------------------")
torch.save(model, checkpoint_path)

print("-----------------------Evaluation Metrics-------------------------------------")
model = torch.load(checkpoint_path)
