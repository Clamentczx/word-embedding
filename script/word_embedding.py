import collections
import itertools
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import sys
from classifier import Trainer
from data_reader import Data_reader

class BiLSTM(torch.nn.Module):
    """
    This is a implementation of BiLSTM network.  
    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

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
        _,word_to_index, _ = Word_embedding(self.corpus).get_vocab()

        all_sentence_tensor = []
        for sentence in self.corpus:
            sentence_vector = []     # embeddings of each words in a sentence
            for word in sentence.split():
                if(word in word_to_index):
                    index = word_to_index[word]
                    word_embedding = self.embedding[index].tolist()
                    sentence_vector.append(word_embedding)
                else:
                    index = word_to_index['<unk>']
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
            if(word in word_to_index):
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

def getCorpus(filetype,MAX_VOCAB_SIZE):
    if filetype == 'dev':
        filepath = './data/dev.txt'
    elif filetype == 'test':
        filepath = './data/test.txt'
    elif filetype == 'origin':
        filepath = './data/train_origin.txt'
    else:
        filepath = './data/train.txt'
   
    corpus=[]
    result=""
    with open(filepath, "r") as f:
        txt1=f.readline()
        while txt1:
            y = txt1.split(" ", 1)
            corpus.append(y[1].replace("\n", ""))
            txt1=f.readline()
    f.close()

    """
    stop_words=[]
    # stop words list
    s=open("./stopwords.txt","r")
    txt1=s.readline()
    while txt1:
        stop_words.append(txt1.replace("\n", ""))
        txt1=s.readline()
    s.close()
    
    #remove stop words
    for i in range(len(corpus)):
        final_list = [word for word in corpus[i].lower().split() if word not in stop_words]
        final_string = ' '.join(final_list)
        corpus[i]=final_string  
    """

    for sentence in corpus:
        result=result+sentence
    
    text = result.lower().split()
    text = text[: min(len(text),len(text) )]
    vocab_dict = dict(collections.Counter(text).most_common(MAX_VOCAB_SIZE - 1))
    vocab_dict['<unk>'] = len(text) - sum(list(vocab_dict.values()))
    idx_to_word = list(vocab_dict.keys())
    word_to_idx = {word:ind for ind, word in enumerate(idx_to_word)}
    word_counts = np.array(list(vocab_dict.values()), dtype=np.float32)
    word_freqs = word_counts / sum(word_counts)
    

    return text, idx_to_word, word_to_idx, word_counts, word_freqs


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
        MAX_VOCAB_SIZE=2500
        text,index_to_word,word_to_index,_,_=getCorpus('origin', MAX_VOCAB_SIZE)
        
        return text,word_to_index, index_to_word

    def get_indices(self):
        """
        Gets a list of indices of corpus.
        :return: indices
        :rtype: list
        """
        _,word_to_index, _ = self.get_vocab()
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
    _,word_to_index, _ = Word_embedding(self.corpus).get_vocab()
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
    _,word_to_index, _ = Word_embedding(self.corpus).get_vocab()
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


def buildCooccuranceMatrix(text, word_to_idx,WINDOW_SIZE):
    vocab_size = len(word_to_idx)
    maxlength = len(text)
    text_ids = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text]
    cooccurance_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    print("Co-Matrix consumed mem:%.2fMB" % (sys.getsizeof(cooccurance_matrix)/(1024*1024)))
    for i, center_word_id in enumerate(text_ids):
        window_indices = list(range(i - WINDOW_SIZE, i)) + list(range(i + 1, i + WINDOW_SIZE + 1))
        window_indices = [i % maxlength for i in window_indices]
        window_word_ids = [text_ids[index] for index in window_indices]
        for context_word_id in window_word_ids:
            cooccurance_matrix[center_word_id][context_word_id] += 1
        if (i+1) % 1000000 == 0:
            print(">>>>> Process %dth word" % (i+1))
    print(">>>>> Save co-occurance matrix completed.")
    return cooccurance_matrix

def buildWeightMatrix(co_matrix):
    xmax = 100.0
    weight_matrix = np.zeros_like(co_matrix, dtype=np.float32)
    print("Weight-Matrix consumed mem:%.2fMB" % (sys.getsizeof(weight_matrix) / (1024 * 1024)))
    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            weight_matrix[i][j] = math.pow(co_matrix[i][j] / xmax, 0.75) if co_matrix[i][j] < xmax else 1
        if (i+1) % 1000 == 0:
            print(">>>>> Process %dth weight" % (i+1))
    print(">>>>> Save weight matrix completed.")
    return weight_matrix

class WordEmbeddingDataset(DataLoader):
    def __init__(self, co_matrix, weight_matrix):
        self.co_matrix = co_matrix
        self.weight_matrix = weight_matrix
        self.train_set = []

        for i in range(self.weight_matrix.shape[0]):
            for j in range(self.weight_matrix.shape[1]):
                if weight_matrix[i][j] != 0:
                    self.train_set.append((i, j))   

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, index):
        (i, j) = self.train_set[index]
        return i, j, torch.tensor(self.co_matrix[i][j], dtype=torch.float), self.weight_matrix[i][j]

class GloveModelForBGD(torch.nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        

        self.v = torch.nn.Embedding(vocab_size, embed_size)
        self.w = torch.nn.Embedding(vocab_size, embed_size)
        self.biasv = torch.nn.Embedding(vocab_size, 1)
        self.biasw = torch.nn.Embedding(vocab_size, 1)
        

        initrange = 0.5 / self.embed_size
        self.v.weight.data.uniform_(-initrange, initrange)
        self.w.weight.data.uniform_(-initrange, initrange)

    def forward(self, i, j, co_occur, weight):

        vi = self.v(i)	
        wj = self.w(j)
        bi = self.biasv(i)
        bj = self.biasw(j)

        similarity = torch.mul(vi, wj)
        similarity = torch.sum(similarity, dim=1)

        loss = similarity + bi + bj - torch.log(co_occur)
        loss = 0.5 * weight * loss * loss

        return loss.sum().mean()

    def gloveMatrix(self):      
        return self.v.weight.data.numpy() + self.w.weight.data.numpy()
    
 
    def train():
        EMBEDDING_SIZE = 300		#300个特征
        MAX_VOCAB_SIZE = 2500	#词汇表大小为2000个词语
        WINDOW_SIZE = 5			#窗口大小为5

        NUM_EPOCHS = 10			#迭代10次
        BATCH_SIZE = 256			#一批有10个样本
        LEARNING_RATE = 0.1
        text, idx_to_word, word_to_idx, word_counts, word_freqs = getCorpus('origin', MAX_VOCAB_SIZE)    

        co_matrix = buildCooccuranceMatrix(text, word_to_idx,WINDOW_SIZE)    #构建共现矩阵
        weight_matrix = buildWeightMatrix(co_matrix)             #构建权重矩阵
        dataset = WordEmbeddingDataset(co_matrix, weight_matrix) #创建dataset
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        model = GloveModelForBGD(MAX_VOCAB_SIZE, EMBEDDING_SIZE) #创建模型
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE) #选择Adagrad优化器


        epochs = NUM_EPOCHS
        iters_per_epoch = int(dataset.__len__() / BATCH_SIZE)
        total_iterations = iters_per_epoch * epochs
        print("Iterations: %d per one epoch, Total iterations: %d " % (iters_per_epoch, total_iterations))

        for epoch in range(epochs):
            loss_print_avg = 0
            iteration = iters_per_epoch * epoch
            for i, j, co_occur, weight in dataloader:
                iteration += 1
                optimizer.zero_grad()   #每一批样本训练前重置缓存的梯度
                loss = model(i, j, co_occur, weight)    #前向传播
                loss.backward()     #反向传播
                optimizer.step()    #更新梯度
                loss_print_avg += loss.item()
        return model.gloveMatrix()
    

class SelfAttentionBiLSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bilstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attention = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_seq):
        output, _ = self.bilstm(input_seq)
        attn_weights = torch.softmax(self.attention(output))
        sentence_vector = torch.sum(attn_weights * output)
        return sentence_vector




 




if __name__ == "__main__":


    gl_embedding = GloveModelForBGD.train()
    input_shape = (None, 300)  # 300-dimensional word vectors

    # Create an instance of the model
    model = SelfAttentionBiLSTMEncoder(input_dim=300, hidden_dim=128, num_layers=1)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    X_train = gl_embedding
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, output)  # use the output as the target
        loss.backward()
        optimizer.step()
     

    
   
    train_data, dev_data, test_data = Data_reader('data').get_data()
    """
    gl_sent_rep_BOW_train = Bow(train_data[2], gl_embedding).get_sentence_weight()
    gl_sent_rep_BOW_dev = Bow(dev_data[2], gl_embedding).get_sentence_weight()
    gl_sent_rep_BOW_test = Bow(test_data[2], gl_embedding).get_sentence_weight()

    gl_bow_trainer = Trainer(train_label=train_data[0], train_sentence=gl_sent_rep_BOW_train, 
                         dev_label=dev_data[0], dev_sentence=gl_sent_rep_BOW_dev, 
                         test_label=test_data[0], test_sentence=gl_sent_rep_BOW_test, 
                         input_size=300, hidden_size=800, batch_size=100, learning_rate=1e-2, num_epochs=10)
    
    gl_bow_trainer.train()
    gl_bow_trainer.test()
    """

    #gl_sent_rep_BiLSTM_train = BiLSTMWrapper(train_data[2], gl_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap() 
    #gl_sent_rep_BiLSTM_dev = BiLSTMWrapper(dev_data[2], gl_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()
    #gl_sent_rep_BiLSTM_test = BiLSTMWrapper(test_data[2], gl_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap() 
    
    """
    gl_bilstm_trainer = Trainer(train_label=train_data[0], train_sentence=gl_sent_rep_BiLSTM_train, 
                         dev_label=dev_data[0], dev_sentence=gl_sent_rep_BiLSTM_dev, 
                         test_label=test_data[0], test_sentence=gl_sent_rep_BiLSTM_test, 
                         input_size=300, hidden_size=720, batch_size=128, learning_rate=1e-2, num_epochs=10)
    
    gl_bilstm_trainer.train()
    gl_bilstm_trainer.test()
    """




    