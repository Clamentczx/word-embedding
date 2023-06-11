import torch
import torch.nn as nn
from word_embedding import Word_embedding


# BiLSTM model 
# Define BiLSTM model
class BiLSTM(nn.Module):
    """
    This is a implementation of BiLSTM network.  
    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(self.device)
        out, (hidden, cell) = self.lstm(x, (h0, c0))

        out = (out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]) / 2
        
        return out[:, -1, :] # Return last hidden state of each direction



class BiLSTMWrapper:
    """
    This is a wrapper class of BiLSTM class.
    """
    def __init__(self, corpus, embedding, input_dim=256, hidden_dim=48, num_layers=2):
        self.corpus = corpus
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = embedding

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def get_sentence_tensor(self):
        """
        Gets sentence tensor of BiLSTM algorithm object.
        """
        word_to_index, _ = Word_embedding(self.corpus).get_vocab()

        all_sentence_tensor = []
        for sentence in self.corpus:
            sentence_vector = []     # embeddings of each words in a sentence
            for word in sentence.split():
                index = word_to_index[word]
                word_embedding = self.embedding[index].tolist()
                sentence_vector.append(word_embedding)
            all_sentence_tensor.append(torch.tensor(sentence_vector).to(self.device))       # list of tensors, each of them is the tensor stack of a words in a sentence.
        return all_sentence_tensor
        
    def wrap(self):
        """
        Wraps the BiLSTM class and initialize it with parameters. 
        """
        model = BiLSTM(self.input_dim, self.hidden_dim, self.num_layers)        # BiLSTM object
        model.to(self.device)
        all_sentence_tensor = self.get_sentence_tensor()

        result = []

        for sentence_tensor in all_sentence_tensor:
            sentence_vector = model(sentence_tensor.unsqueeze(0))

            result.append(sentence_vector.squeeze())

        return result

# BOW model 
class Bow:
  def __init__(self, corpus, embedding):
      self.corpus = corpus
      self.embedding = embedding
  
  def get_sentence_weight(self):
    text,word_to_index, _ = Word_embedding(self.corpus).get_vocab()

    all_sentence_tensor = []
    for sentence in self.corpus:
        sentence_vector = []     # embeddings of each words in a sentence
        for word in sentence.split():
            if(word in text):
                index = word_to_index[word]
                word_embedding = self.embedding[index].tolist()
                sentence_vector.append(word_embedding)
            else:
                index = word_to_index['<unk>']
                word_embedding = self.embedding[index].tolist()
                sentence_vector.append(word_embedding)
        avg = [sum(col) / float(len(col)) for col in zip(*sentence_vector)]
        all_sentence_tensor.append(torch.tensor(avg))       # list of tensors, each of them is the tensor stack of average weight in a sentence.
    return all_sentence_tensor
  
