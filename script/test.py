from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import torch

from classifier import Trainer

from data_reader import Data_reader

train_data, dev_data, test_data = Data_reader('data').get_data()

train_tokenized_data = [word_tokenize(sentence.lower()) for sentence in train_data[2]]
dev_tokenized_data = [word_tokenize(sentence.lower()) for sentence in dev_data[2]]
test_tokenized_data = [word_tokenize(sentence.lower()) for sentence in test_data[2]]

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(train_data[2])]

model = Doc2Vec(tagged_data, vector_size=300, window=2, min_count=1, workers=4, epochs=100)

train_sentence_vectors = [torch.from_numpy(model.infer_vector(sentence)) for sentence in train_tokenized_data]
dev_sentence_vectors = [torch.from_numpy(model.infer_vector(sentence)) for sentence in dev_tokenized_data]
test_sentence_vectors = [torch.from_numpy(model.infer_vector(sentence)) for sentence in test_tokenized_data]


gensim_trainer = Trainer(train_label=train_data[0], train_sentence=train_sentence_vectors, 
                         dev_label=dev_data[0], dev_sentence=dev_sentence_vectors, 
                         test_label=test_data[0], test_sentence=test_sentence_vectors, 
                         input_size=300, hidden_size=720, batch_size=128, learning_rate=7e-3, num_epochs=10)
    
gensim_trainer.train()
gensim_trainer.test()


