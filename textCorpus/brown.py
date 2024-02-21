import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords


'''
Output : Nested list token words
         mapping : Key : Word  Value : Index
         reverse_mapping : Key : Index , Value : Word
'''

           
def dataset() :
    print("Importing Brown dataset")
    # print("Total length of words : ", len(brown.words()))
    # print("50 sample words from dataset : " , brown.words())
    nltk.download('brown')
    sentences = brown.sents() 
    # print("Total sentences : ", len(sentences))
    dataset = data_preprocessing(dataset= sentences)
    # print("Sample dataset sentence :  " , dataset[100])
    mapping , words  = get_vocab(dataset= dataset)
    dataset_tokens = convert_word_to_index(dataset= dataset, mapping= mapping)
    reverse_mapping = reverse_dict(mapping= mapping)
    # print("Sample dataset sentence token :  " , dataset_tokens[100])

    return dataset_tokens, mapping, reverse_mapping
    

# Input : Nested of list of sentences
# Output : Nested of list of sentences

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

# Input : Nested of list of sentences
# Output : Dict of words where key : word and value : word index and vocbulary list 
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
        mapping[words[i]] = i+1 

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
 
'''
Driver code logic 
Just call dataset function to get all the sentence in |V| length where each word is replace by corresponding index 

dataset_token, mapping, reverse_mapping = dataset()
print(dataset_token[100])
idx = 812
print("Word at index : ", idx , " - ", reverse_mapping[idx])
print("Index for word : ", reverse_mapping[idx],  " - " , idx)


'''