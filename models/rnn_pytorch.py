# Import  libraries
import sys
sys.path.insert(1,'../')
import torch
from torch import nn
import torch.nn.functional as torchFunctional
import textCorpus.brown as brown
from torchsummary import  summary
from sklearn.model_selection import train_test_split


'''
Input  : Sentence (List of tokens for a sentence)
         mapping : Key : Word  Value : Index
         reverse_mapping : Key : Index , Value : Word

Output : Return the nested list of token in one hot vector form 
'''



def convert_dataset_to_tensor(sentence: list, mapping: dict, reverse_mapping: dict):

    num_classes = len(list(mapping.keys())) + 1
    # print("Vocabulary Length : ", num_classes)
    # print("Input Sentence : " , sentence)
    sentence_tensor = torch.tensor(sentence)
    sentence_token_one_hot = torchFunctional.one_hot(sentence_tensor, num_classes=num_classes)
    # print("Input One Hot : ", sentence_token_one_hot)
    # print("Input One Hot Shape : ", sentence_token_one_hot.shape)
    return sentence_token_one_hot




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
        print(e.shape)
        h1 = self.weight(e)
        print(h1.shape)
        h2 = self.u(hidden_state)
        print(h2.shape)
        h = torch.add(h1, h2)
        print(h.shape)
        out = self.v(h)
        print(out.shape)
        out = torch.softmax(out, dim = 1 )
        print(out.shape)
        out.reshape(-1)
        print(out.shape)
        return out, h

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))


dataset_token, mapping, reverse_mapping = brown.dataset()
sentence_token_one_hot = convert_dataset_to_tensor(sentence= dataset_token[0], mapping= mapping, reverse_mapping= reverse_mapping)
print("Starting One hot conversion")
dataset_one_hot = []
# length = len(dataset_token)
length = 100
for i in range(length) :
    dataset_one_hot.append(convert_dataset_to_tensor(sentence = dataset_token[i], mapping= mapping, reverse_mapping= reverse_mapping))
print("Completed One hot conversion")
print("Dataset : ", len(dataset_one_hot))
print("dataset ------------> ", dataset_one_hot[0].shape)

print("Starting Train Test Split")
# Splitting dataset into train, test, validation
train_idx, test_idx = train_test_split(
    range(len(dataset_one_hot)),
    test_size=0.1,
    shuffle=True,
)

train_dataset = [
    (dataset_one_hot[i])
    for i in train_idx
]

test_dataset = [
    (dataset_one_hot[i])
    for i in test_idx
]

print("Train Split : ", len(train_dataset))
print("dataset ------------> ", train_dataset[0].shape)
print("Test Split : ", len(test_dataset))
print("dataset ------------> ", test_dataset[0].shape)


input_size = sentence_token_one_hot.shape[1]
embedding_size = 300
hidden_size = 256
output_size = sentence_token_one_hot.shape[1]
learning_rate = 0.001
epochs = 11

model = RNN(input_size = input_size, embedding_size= embedding_size, hidden_size= hidden_size, output_size= output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
print("Model Summary")
print(summary(model, input_size = [(1, input_size), (1, hidden_size)] , batch_size= -1))

for epoch in range(epochs) :
    for i in range(2) : # len(train_dataset)
        hidden_state = model.init_hidden()
        print("hedskfjsdf", hidden_state.shape)
        sentence = train_dataset[i]
        print(sentence)
        print(type(sentence))
        print(sentence.shape)
        for i in range(sentence.shape[0]-1) :
            input_vector = sentence[i].float().reshape(1, -1)
            output_vector = sentence[i+1].float().reshape(1, -1)
            print("Input Shape : ", input_vector.shape)
            print("Output Shpae : ", output_vector.shape)
            output_hat, hidden_state = model(input_vector, hidden_state)
            curr_loss = criterion(output_hat, output_vector)
            total_loss = curr_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(epoch)

print("Kuch hua hai")





