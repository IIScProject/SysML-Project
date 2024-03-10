import os
import sys
import torch
from torch import nn
import textCorpus.brown as brown
from torchsummary import summary
import utilities as utilities

sys.path.insert(1, '../')

'''
Input  : Sentence (List of tokens for a sentence)
         mapping : Key : Word  Value : Index
         reverse_mapping : Key : Index , Value : Word

Output : Return the nested list of token in one hot vector form 
'''


class LSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_state_size, context_state_size, output_size):
        super(LSTM, self).__init__()
        '''
        Input Size : |V| X 1
        Embedding Size : d X 1
        Embedding Matrix : d X V 
        Input Size == Output Size

        '''
        self.input_size = input_size  # size of input word x ( = Length of Vocabulary (input is OneHot) )
        self.embedding_size = embedding_size  # size of embedding vector for each input word x
        self.hidden_state_size = hidden_state_size  # size of hidden state vector
        self.context_state_size = context_state_size  # size of context state vector
        self.output_size = output_size  # size of output vector ( = Length of Vocabulary )

        self.embedding_layer = nn.Linear(in_features=input_size, out_features=embedding_size, bias=False)

        self.Wf = nn.Linear(in_features=embedding_size, out_features=hidden_state_size, bias=False)
        self.Uf = nn.Linear(in_features=hidden_state_size, out_features=hidden_state_size, bias=False)
        self.forget_gate = nn.Sigmoid()

        self.Wg = nn.Linear(in_features=embedding_size, out_features=hidden_state_size, bias=False)
        self.Ug = nn.Linear(in_features=hidden_state_size, out_features=hidden_state_size, bias=False)
        self.extract = nn.Tanh()
        self.output = nn.Linear(in_features=hidden_state_size, out_features=output_size, bias=False)

        self.Wi = nn.Linear(in_features=embedding_size, out_features=hidden_state_size, bias=False)
        self.Ui = nn.Linear(in_features=hidden_state_size, out_features=hidden_state_size, bias=False)
        self.input_gate = nn.Sigmoid()  # acts as a mask for output, deciding what to add (input) to the context

        self.Wo = nn.Linear(in_features=embedding_size, out_features=hidden_state_size, bias=False)
        self.Uo = nn.Linear(in_features=hidden_state_size, out_features=hidden_state_size, bias=False)
        self.output_gate = nn.Sigmoid()

    def lstm_cell(self, input, hidden_state, context_state):
        input_embedding = self.embedding_layer(input)  # generates embedding of input
        f = self.forget_gate(torch.add(self.Wf(input_embedding), self.Uf(hidden_state)))
        g = self.extract(torch.add(self.Wg(input_embedding), self.Ug(hidden_state)))
        i = self.input_gate(torch.add(self.Wi(input_embedding), self.Ui(hidden_state)))
        o = self.output_gate(torch.add(self.Wo(input_embedding), self.Uo(hidden_state)))

        k = torch.mul(context_state, f)
        j = torch.mul(g, i)

        context_state = torch.add(j, k)

        t = nn.Tanh()
        hidden_state = torch.mul(o, t(context_state))

        output = self.output(g)
        return output, hidden_state, context_state  # g is output

    def init_hidden_state(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_state_size))

    def init_context_state(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_state_size))

    def forward(self, input, hidden_state, context_state):
        output_hat_ls = []
        for sequence_length_idx in range(input.shape[1]):
            output_hat_vector, hidden_state, context_state = self.lstm_cell(input[:, sequence_length_idx, :], hidden_state, context_state)
            output_hat_ls.append(output_hat_vector)

        output_hat_stack = torch.stack(output_hat_ls, dim=1)
        return output_hat_stack


def train_model_cpu_gpu(model, criterion, optimizer, epochs, mini_batch_size, train_dataset, test_dataset, mapping,
                        device, checkpoint_path):
    # Define Checkpointing parameters
    checkpoint_frequency = 50  # save checkpoint after every 100 minibatches
    min_loss_improvement = 0.0001  # Minimum improvement in loss to save a new checkpoint
    best_loss = float('inf')

    train_dataset = torch.tensor(train_dataset)
    test_dataset = torch.tensor(test_dataset)
    train_data_loader, test_data_loader = brown.transform_dataLoader(train_dataset=train_dataset,
                                                                     test_dataset=test_dataset,
                                                                     batch_size=mini_batch_size)

    print("----------------Training Model---------------------------")
    for epoch in range(epochs):
        for batch_idx, (data) in enumerate(train_data_loader):
            model.train()
            hidden_state = model.init_hidden_state()
            context_state = model.init_context_state()

            data = torch.tensor(data)
            data_onehot = torch.nn.functional.one_hot(data, num_classes=len(list(mapping.keys())))
            data_onehot = data_onehot.float()
            data_onehot.to(device)
            input_vector = data_onehot[:, :-1, :]
            output_vector = data_onehot[:, 1:, :]

            hidden_state = hidden_state.to(device)
            context_state = context_state.to(device)
            input_vector = input_vector.to(device)
            output_vector = output_vector.to(device)

            output_hat_vector = model(input=input_vector, hidden_state=hidden_state, context_state=context_state)
            loss = criterion(output_vector, output_hat_vector)
            loss += utilities.l2_loss(model, lambda_l2=0.01)  # L2 regularization
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
hidden_state_size = 256
context_state_size = 256
output_size = input_size
learning_rate = 0.01
epochs = 100
mini_batch_size = 1024

print("----------------Creating LSTM Pytorch Model-----------------------")
model = LSTM(input_size=input_size, embedding_size=embedding_size,
             hidden_state_size=hidden_state_size, context_state_size=context_state_size, output_size=output_size)
# Loading model from checkpoint if exists
checkpoint_dir = "../checkpoints/lstm_pytorch/"
checkpoint_path = os.path.join(checkpoint_dir, f"lstm_pytorch.pth")
if os.path.exists(checkpoint_path):
    model = torch.load(checkpoint_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("-------------------------Model Summary----------------------------------------")
# print(summary(model, input_size=[(1, input_size), (1, hidden_state_size), (1, context_state_size)], batch_size=-1))

model = train_model_cpu_gpu(model=model, criterion=criterion, optimizer=optimizer, epochs=epochs,
                            mini_batch_size=mini_batch_size, train_dataset=train_dataset, test_dataset=test_dataset,
                            mapping=mapping, device=device, checkpoint_path=checkpoint_path)

print("------------------------Saving Model Details----------------------------------")
torch.save(model, checkpoint_path)

print("-----------------------Evaluation Metrics-------------------------------------")
model = torch.load(checkpoint_path)
