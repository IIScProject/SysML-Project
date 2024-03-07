import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as torchFunctional


'''
Output : Nested list token words
         mapping : Key : Word  Value : Index
         reverse_mapping : Key : Index , Value : Word
'''

           
def dataset() :
    print("Importing Brown dataset")
    nltk.download('brown')
    sentences = brown.sents() 
    dataset = data_preprocessing(dataset= sentences)
    dataset = converting_dataset_fixed_sequence_length(dataset= dataset)
    mapping , words  = get_vocab(dataset= dataset)
    dataset_tokens = convert_word_to_index(dataset= dataset, mapping= mapping)
    reverse_mapping = reverse_dict(mapping= mapping)

    return dataset_tokens, mapping, reverse_mapping
    
'''
Input : Nested of list of sentences
Output : Nested of list of sentences
'''

def converting_dataset_fixed_sequence_length(dataset, sequence_length = 5) :
    converted_dataset = []
    for i in range(len(dataset)) :
        if len(dataset) <= 5 :
            converted_dataset.append(dataset[i])
        else :
            for j in range(len(dataset[i]) - sequence_length) :
                converted_dataset.append(dataset[i][j:j+sequence_length])
                j += 2
    return converted_dataset


def data_preprocessing(dataset : list) :
    # Lower case conversion
    output = [] 
    for i in range(len(dataset)) : 
        sentence = []
        for j in range(len(dataset[i])) : 
            word = dataset[i][j]
            word = word.lower()
            sentence.append(word)
        output.append(sentence)

    return output

'''
Input : Nested of list of sentences
Output : Dict of words where key : word and value : word index and vocbulary list 
'''

def get_vocab(dataset : list) : 
    
    mapping = {} # Key : Word maps to value : frequency 
    for sentence in dataset : 
        for word in sentence : 
            if word not in mapping : 
                mapping[word] = 1 
            else : 
                mapping[word] += 1 

    words = list(mapping.keys()) 
    mapping = {} # Key : Word and value : Index 
    for i in range(len(words)) : 
        mapping[words[i]] = i

    return mapping , words 


# Input : Nested list of sentence,  Dictionary mapping 
# Output : Dataset tokens
def convert_word_to_index(dataset : list, mapping : dict) : 

    for i in range(len(dataset)) : 
        for j in range(len(dataset[i])) : 
            word = dataset[i][j]
            dataset[i][j] = mapping[word]
    
    return dataset 


# Input : Key : Word , Value : Index 
# Output : Key : Index, value : word
def reverse_dict(mapping : dict) :

    reverse_map = {} 
    key_ls = list(mapping.keys()) 
    for i in range(len(key_ls)) : 
        word = key_ls[i]
        idx = mapping[word]
        reverse_map[idx] = word

    return reverse_map


def train_test_slit(dataset, test_size = 0.1) :
     train_idx, test_idx = train_test_split(
         range(len(dataset)),
         test_size=0.1,
         shuffle=True,
     )

     train_dataset = [
         (dataset[i])
         for i in train_idx
     ]

     test_dataset = [
         (dataset[i])
         for i in test_idx
     ]

     print("Train Split : ", len(train_dataset))
     print("Test Split : ", len(test_dataset))
     return train_dataset, test_dataset

def convert_sentence_to_one_hot_tensor(sentence: list, mapping: dict):

    num_classes = len(list(mapping.keys()))
    sentence_tensor = torch.tensor(sentence)
    sentence_token_one_hot = torchFunctional.one_hot(sentence_tensor, num_classes=num_classes)
    sentence_token_one_hot = sentence_token_one_hot.float()
    return sentence_token_one_hot


'''
Driver code logic 
Just call dataset function to get all the sentence in |V| length where each word is replace by corresponding index 

dataset_token, mapping, reverse_mapping = dataset()
print(dataset_token[100])
idx = 812
print("Word at index : ", idx , " - ", reverse_mapping[idx])
print("Index for word : ", reverse_mapping[idx],  " - " , idx)
'''