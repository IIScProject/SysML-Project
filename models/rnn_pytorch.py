# Import  libraries
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

class RNN(nn.Module) :
    def __init__(self, input_size , embedding_size, hidden_size, output_size):
        super(RNN, self).__init__()
        '''
        Input Size : 1 * |V| 
        Embedding Size : d 
        Embedding Matrix : |V| * d 
        
        
        '''
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Linear(in_features= input_size, out_features= embedding_size, bias= False )
        self.weight = nn.Linear(in_features= embedding_size, out_features= hidden_size, bias= False)
        self.u = nn.Linear(in_features= hidden_size, out_features= hidden_size, bias= False)
        self.v = nn.Linear(in_features= hidden_size, out_features= input_size, bias= False)

    def forward(self, input, hidden_state):
        e = self.embedding(input)
        h1 = self.weight(e)
        h2 = self.u(hidden_state)
        h = torch.add(h1, h2)
        out = self.v(h)
        out = torch.softmax(out, dim = 1 )
        out.reshape(-1)
        return out, h

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))




dataset, mapping, reverse_mapping = brown.dataset()
train_dataset, test_dataset = brown.train_test_slit(dataset)
print("Starting Train Test Split")
# Splitting dataset into train, test, validation

input_size = len(mapping)
embedding_size = 300
hidden_size = 256
output_size = input_size
learning_rate = 0.01
epochs = 11
mini_batch_size = 1024

model = RNN(input_size = input_size, embedding_size= embedding_size, hidden_size= hidden_size, output_size= output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print("Model Summary")
print(summary(model, input_size = [(1, input_size), (1, hidden_size)] , batch_size= -1))

for epoch in range(epochs) :
    model.train()
    mini_batches = len(train_dataset) // mini_batch_size
    idx = 0
    for mini_batch in range(mini_batches) :
        mini_batch_loss = 0
        hidden_state = model.init_hidden()
        total_dividend = 0
        for _ in range(mini_batch_size) :
            sentence = train_dataset[idx]
            idx += 1
            idx = idx % len(train_dataset)
            # Converting two one-hot vectors
            sentence = brown.convert_sentence_to_one_hot_tensor(sentence= sentence, mapping= mapping)
            sentence_loss = 0
            total_dividend += len(sentence) - 1
            for sentence_idx in range(len(sentence) - 1) :
                input_vector = sentence[sentence_idx].float().reshape(1, -1)
                output_vector = sentence[sentence_idx + 1].float().reshape(1, -1)
                output_hat, hidden_state = model(input_vector, hidden_state)
                loss = criterion(output_vector, output_hat)
                if sentence_loss == 0 :
                    sentence_loss = loss
                else :
                    sentence_loss += loss
            if mini_batch_loss == 0 :
                mini_batch_loss = sentence_loss
            else :
                mini_batch_loss += sentence_loss
        optimizer.zero_grad()
        mini_batch_loss /= total_dividend
        mini_batch_loss += utilities.l2_loss(model, lambda_l2= 0.01)
        mini_batch_loss.backward()
        optimizer.step()
        print(f"Epoch : {epoch + 1}, Min-batch : {mini_batch + 1}, training-loss : {mini_batch_loss}")






