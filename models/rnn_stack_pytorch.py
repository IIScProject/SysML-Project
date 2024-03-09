# Import  libraries
import os
import sys
import torch
from torch import nn
import textCorpus.brown as brown
from torchsummary import  summary
import utilities as utilities
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

def train_model_cpu_gpu_stack(model, criterion, optimizer, epochs, mini_batch_size,
                              train_dataset, test_dataset, mapping, device, checkpoint_path, stack_length) :

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
            output_hat_vector = model(input = input_vector, hidden_states = hidden_state, stack_length = stack_length)
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
                    torch.save(model, checkpoint_path)  # Saving Model
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
mini_batch_size = 1024
stack_length = 4

print("----------------Creating RNN Pytorch Stack Model-----------------------")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)
model = RNN_stack(input_size = input_size, embedding_size= embedding_size,
            hidden_size= hidden_size, output_size= output_size, stack_length= stack_length)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Loading model from checkpoint if exists
checkpoint_dir = "../checkpoints/rnn_pytorch/"
checkpoint_path = os.path.join(checkpoint_dir, f"rnn_pytorch_stack.pth")
if os.path.exists(checkpoint_path) :
    model = torch.load(checkpoint_path)


print("-------------------------Model Summary----------------------------------------")
# print(summary(model, input_size = [(1, 4, 49512), (1, 4, 256), 4] , batch_size= -1)) # Need to work on it
model = train_model_cpu_gpu_stack(model= model, criterion= criterion, optimizer= optimizer, epochs= epochs,
                            mini_batch_size= mini_batch_size, train_dataset= train_dataset, test_dataset= test_dataset,
                            mapping= mapping, device= device, checkpoint_path = checkpoint_path,
                            stack_length= stack_length)

print("------------------------Saving Model Details----------------------------------")
torch.save(model, checkpoint_path)

print("-----------------------Evaluation Metrics-------------------------------------")
model = torch.load(checkpoint_path)