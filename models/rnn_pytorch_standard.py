# Import  libraries
import os
import sys
import torch
from torch import nn
sys.path.insert(1,'../')
import textCorpus.brown as brown
from torchsummary import  summary
import utilities as utilities

'''
Input  : Sentence (List of tokens for a sentence)
         mapping : Key : Word  Value : Index
         reverse_mapping : Key : Index , Value : Word

Output : Return the nested list of token in one hot vector form 
'''

class RNNStandard(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers):
        super(RNNStandard, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden):
        print("Input Shape : ", input.shape)
        print("Hidden Shape : ", hidden.shape)
        embedded = nn.Linear(in_features= input.shape[-1], out_features= embedding_size).to(device)(input)
        print("Embedded Shape : ", embedded.shape)
        output, hidden = self.rnn(embedded, hidden)
        print("Output Shape : ", output.shape)
        output = nn.Linear(in_features= output.shape[-1], out_features= input_size).to(device)(output)
        output = nn.functional.log_softmax(output, dim=1)  # Applying log_softmax to get log probabilities
        print("Output Shape : ", output.shape)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size)


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
            hidden_state = model.init_hidden(batch_size= mini_batch_size)
            print("Hidden State : ", hidden_state.shape)
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
            print("Loss", loss)
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
mini_batch_size = 1
num_layers = 1

print("----------------Creating RNN Pytorch Model-----------------------")
model = RNNStandard(input_size = input_size, hidden_size = hidden_size,
                    embedding_size = embedding_size, num_layers = num_layers)
# Loading model from checkpoint if exists
checkpoint_dir = "../checkpoints/rnn_pytorch/"
checkpoint_path = os.path.join(checkpoint_dir, f"rnn_pytorch_standard.pth")
if os.path.exists(checkpoint_path) :
    model = torch.load(checkpoint_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("-------------------------Model Summary----------------------------------------")
# print(summary(model, input_size = [(4, 49512), (1, 256)]))
print("RNN Model : \n", model)
model = train_model_cpu_gpu(model= model, criterion= criterion, optimizer= optimizer, epochs= epochs,
                            mini_batch_size= mini_batch_size, train_dataset= train_dataset, test_dataset= test_dataset,
                            mapping= mapping, device= device, checkpoint_path= checkpoint_path)

print("------------------------Saving Model Details----------------------------------")
torch.save(model, checkpoint_path)

print("-----------------------Evaluation Metrics-------------------------------------")
model = torch.load(checkpoint_path)

