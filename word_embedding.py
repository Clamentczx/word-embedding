import collections
import itertools
import torch
from torch.utils.data import DataLoader
import numpy as np

class Word_embedding:
    """
    A class that gets corpus and its threshold, returns word index dictionary and indices.
    """
    def __init__(self, corpus, threshold=0):
        self.corpus = corpus
        self.threshold = threshold

    def get_vocab(self):
        """
        Gets word to index and index to word dictionary.
        :return: word_to_index, index_to_word dictionary
        :rtype: dict
        """
        words = list(itertools.chain.from_iterable([sentence.split() for sentence in self.corpus]))
        # count the frequencies of each word
        word_counts = collections.Counter(words)
        # set vocabulary with word occurances at least 3 times
        vocab = sorted([word for word in word_counts if word_counts[word] >= self.threshold])# assign an index to each word
        word_to_index = {word: index for index, word in enumerate(vocab)}
        index_to_word = {index: word for index, word in enumerate(vocab)}
        return word_to_index, index_to_word

    def get_indices(self):
        """
        Gets a list of indices of corpus.
        :return: indices
        :rtype: list
        """
        word_to_index, _ = self.get_vocab()
        indices=[[word_to_index[word] for word in sentences.split()] for sentences in self.corpus]
        return indices

class Random_init:
  """
  Randomly initialize a tensor that shaped to required number of dimensions.
  """
  def __init__(self, corpus, embedding_dims=128):
    self.corpus = corpus
    self.embedding_dims = embedding_dims

  def get_weight(self):
    """
    Returns the randomly initialized tensor.
    """
    word_to_index, _ = Word_embedding(self.corpus).get_vocab()
    vocab_size = len(word_to_index)
    embedding_dim = self.embedding_dims  
    embeds = torch.nn.Embedding(vocab_size, embedding_dim)
    torch.manual_seed(1)
    return embeds.weight

class Glove:
  """
  This is a class that loads from file of pre-trained weight and transfer it to a tensor.
  """
  def __init__(self, corpus):
    self.corpus = corpus

  def get_weight(self):
    """
    Returns the pre-trained weight tensor.
    """
    word_to_index, _ = Word_embedding(self.corpus).get_vocab()
    vocab_size = len(word_to_index) 

    word_embeddings = {}
    with open('./glove.small.txt', 'r') as files:
      for line in files:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        word_embeddings[word] = vector
    
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_to_index.items():
      if word in word_embeddings:
        embedding_matrix[i] = word_embeddings[word]
      else:
        embedding_matrix[i] = word_embeddings["#UNK#"]

    embedding_layer = torch.nn.Embedding(vocab_size, 300)
    embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))

    return embedding_layer.weight 

class Cbow:
    """
    A class that implements CBOW algorithm. It is used to predict the target word for given context words.  
    """
    def __init__(self, corpus, window=2, embedding_dims=128, batch_size=5, num_epochs=500, learning_rate = 6e-2):
        self.corpus = corpus
        self.window = window
        self.embedding_dims = embedding_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def get_corpus(self):
        """
        Gets corpus of CBOW algorithm object.
        """
        return self.corpus

    def get_data(self):
        """
        Returns a list of sublists with context words and their target word.
        :return: bows
        :rtype: list
        """
        word_to_index, _ = Word_embedding(self.corpus).get_vocab()
        result = []
        for sentence in self.corpus:
            for i in range(self.window, len(sentence.split()) - 2):
                context = [sentence.split()[i - 2], sentence.split()[i - 1],
                        sentence.split()[i + 1], sentence.split()[i + 2]]
                target = sentence.split()[i]
                result.append([[word_to_index[w] for w in context], word_to_index[target]])
        return result
    
    @staticmethod
    def get_input_tensor(tensor, vocab_size, device):
        """
        Gets the torch autograd variable for CBOW algorithm object's input layer.
        """
        size = [*tensor.shape][0]
        inp = torch.zeros(size, vocab_size, device=device).scatter_(1, tensor, 1.)
        return torch.autograd.Variable(inp).float()

class Skip_gram:
    """
    A class that implements skip-gram algorithm. It is used to predict the context word for a given word.  
    """
    def __init__(self, corpus, window=2):
        self.corpus = corpus
        self.window = window

    def get_corpus(self):
        """
        Gets corpus of skip-gram algorithm object.
        """
        return self.corpus

    def get_data(self):
        """
        Returns a list of sublists of context word and its given word.
        :return: n grams.
        :rtype: list
        """
        word_to_index, _ = Word_embedding(self.corpus).get_vocab()
        result = []
        for sentence in self.corpus:
            for i,w in enumerate(sentence.split()):
                inp = w
                for n in range(1,self.window+1):
                    # look back
                    if (i-n)>=0:
                        out = sentence.split()[i-n]
                        result.append([word_to_index[inp], word_to_index[out]])
                    
                    # look forward
                    if (i+n)<len(sentence.split()):
                        out = sentence.split()[i+n]
                        result.append([word_to_index[inp], word_to_index[out]])
        return result
    
    @staticmethod
    def get_input_tensor(tensor, vocab_size, device):
        """
        Gets the torch autograd variable for skip-gram algorithm object's input layer.
        """
        size = [*tensor.shape][0]
        inp = torch.zeros(size, vocab_size, device=device).scatter_(1, tensor.unsqueeze(1), 1.)
        return torch.autograd.Variable(inp).float()
    
class Trainer:
    def __init__(self, algorithm, embedding_dims=256, batch_size=128, num_epochs=320, learning_rate = 6e-2):
        self.algorithm = algorithm
        self.embedding_dims = embedding_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        

    def train(self):
        """
        Trainer for algorithm object.
        """
        
        data = self.algorithm.get_data()

        word_to_index, _ = Word_embedding(self.algorithm.get_corpus()).get_vocab()
        vocab_size = len(word_to_index)

        # move data to gpu
        input = torch.tensor([pair[0] for pair in data], device=self.device)
        output = torch.tensor([pair[1] for pair in data], device=self.device)


        initrange = 0.5 / self.embedding_dims
        W1 = torch.autograd.Variable(torch.randn(vocab_size, self.embedding_dims, device=self.device).uniform_(-initrange, initrange).float(), requires_grad=True) # shape V*H
        W2 = torch.autograd.Variable(torch.randn(self.embedding_dims, vocab_size, device=self.device).uniform_(-initrange, initrange).float(), requires_grad=True) # shape H*V
        print(f'W1 shape is: {W1.shape}, W2 shape is: {W2.shape}')
        
        # define optimizer
        optimizer = torch.optim.Adam([W1, W2], lr = self.learning_rate)

        loss_hist = []
        for epo in range(self.num_epochs):
            for x, y in zip(DataLoader(input, batch_size=self.batch_size), DataLoader(output, batch_size=self.batch_size)):
                # one-hot encode input tensor
                input_tensor = self.algorithm.get_input_tensor(x, vocab_size, self.device) #shape N*V
            
                # simple NN architecture
                h = input_tensor.mm(W1) # shape 1*H
                y_pred = h.mm(W2) # shape 1*V
                
                # define loss function
                loss_f = torch.nn.CrossEntropyLoss() 
                
                #compute loss
                loss = loss_f(y_pred, y)
                
                # zeroing gradients after each iteration
                optimizer.zero_grad()

                # bakpropagation step
                loss.backward()

                # use gradient to perform the optimization
                optimizer.step()

            #lr_decay = 0.99
            """
            if epo%10 == 0:
                learning_rate *= lr_decay
            """
            loss_hist.append(loss)
            print(f'Epoch {epo}, loss = {loss}')
        
        # each column of W1 stores representation for single word
        return W1

    
if __name__ == "__main__":
    pass