import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords


# Output : Nested list token words and mapping for tokens 
def dataset() :
    # print("Importing Brown dataset")
    # print("Total length of words : ", len(brown.words()))
    # print("50 sample words from dataset : " , brown.words())
    sentences = brown.sents() 
    # print("Total sentences : ", len(sentences))
    dataset = data_preprocessing(dataset= sentences)
    # print("Sample dataset sentence :  " , dataset[100])
    mapping, words  = get_vocab(dataset= dataset)
    dataset_tokens = convert_word_to_index(dataset= dataset, mapping= mapping)
    # print("Sample dataset sentence token :  " , dataset_tokens[100])

    return dataset_tokens, mapping
    

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

 
'''
Driver code logic 
Just call dataset function to get all the sentence in |V| length where each word is replace by corresponding index 
'''
dataset_token, mapping = dataset()


'''
One time run python file 
nltk.download('brown')
nltk.download('stopwords')

'''

