import torch
torch.manual_seed(1)
from data_reader import Data_reader
from word_embedding import Skip_gram, Cbow, Trainer, Random_init, Glove
from sentence_representation import BiLSTMWrapper, Bow
from classifier import Trainer

def main():
    train_data, dev_data, test_data = Data_reader('data').get_data()        # each data = [coarse, fine, sentence]
    def train_word_embedding():
        # skip gram 
        sk = Skip_gram(train_data[2], window=2)
        sk_trainer = Trainer(sk, embedding_dims=300, batch_size=128, num_epochs=320, learning_rate=7e-5)
        sk_embedding = sk_trainer.train()      # each column of W1 stores representation for single word

        Data_reader.writer("./", sk_embedding, "word_embeddings_sk")

        # cbow
        cb = Cbow(train_data[2], window=2)
        cb_trainer = Trainer(cb, embedding_dims=300, batch_size=128, num_epochs=320, learning_rate=7e-5)      
        cb_embedding = cb_trainer.train()      # each column of W1 stores representation for single word

        Data_reader.writer("./", cb_embedding, "word_embeddings_cb")

    # comment if training word embedding is not required
    #train_word_embedding()

    # load weight
    sk_embedding = Data_reader.loader("./", "word_embeddings_sk")
    cb_embedding = Data_reader.loader("./", "word_embeddings_cb")

    ra_embedding = Random_init(train_data[2], embedding_dims=300).get_weight()
    gl_embedding = Glove(train_data[2]).get_weight()         # this is a 300 dims embedding

    # BOW
    #sk_sent_rep_BOW_train = Bow(train_data[2], sk_embedding).get_sentence_weight()
    cb_sent_rep_BOW_train = Bow(train_data[2], cb_embedding).get_sentence_weight()
    #ra_sent_rep_BOW_train = Bow(train_data[2], ra_embedding).get_sentence_weight()
    gl_sent_rep_BOW_train = Bow(train_data[2], gl_embedding).get_sentence_weight()

    #sk_sent_rep_BOW_dev = Bow(dev_data[2], sk_embedding).get_sentence_weight()
    cb_sent_rep_BOW_dev = Bow(dev_data[2], cb_embedding).get_sentence_weight()
    #ra_sent_rep_BOW_dev = Bow(dev_data[2], ra_embedding).get_sentence_weight()
    gl_sent_rep_BOW_dev = Bow(dev_data[2], gl_embedding).get_sentence_weight()

    #sk_sent_rep_BOW_test = Bow(test_data[2], sk_embedding).get_sentence_weight()
    cb_sent_rep_BOW_test = Bow(test_data[2], cb_embedding).get_sentence_weight()
    #ra_sent_rep_BOW_test = Bow(test_data[2], ra_embedding).get_sentence_weight()
    gl_sent_rep_BOW_test = Bow(test_data[2], gl_embedding).get_sentence_weight()

    # BiLSTM     
    #sk_sent_rep_BiLSTM_train = BiLSTMWrapper(train_data[2], sk_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()
    #cb_sent_rep_BiLSTM_train = BiLSTMWrapper(train_data[2], cb_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()    
    #ra_sent_rep_BiLSTM_train = BiLSTMWrapper(train_data[2], ra_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()    
    gl_sent_rep_BiLSTM_train = BiLSTMWrapper(train_data[2], gl_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap() 

    #sk_sent_rep_BiLSTM_dev = BiLSTMWrapper(dev_data[2], sk_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()
    #cb_sent_rep_BiLSTM_dev = BiLSTMWrapper(dev_data[2], cb_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()    
    #ra_sent_rep_BiLSTM_dev = BiLSTMWrapper(dev_data[2], ra_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()    
    gl_sent_rep_BiLSTM_dev = BiLSTMWrapper(dev_data[2], gl_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap() 

    #sk_sent_rep_BiLSTM_test = BiLSTMWrapper(test_data[2], sk_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()
    #cb_sent_rep_BiLSTM_test = BiLSTMWrapper(test_data[2], cb_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()    
    #ra_sent_rep_BiLSTM_test = BiLSTMWrapper(test_data[2], ra_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap()    
    gl_sent_rep_BiLSTM_test = BiLSTMWrapper(test_data[2], gl_embedding, input_dim=300, hidden_dim=300, num_layers=2).wrap() 

    """
    cb_bow_trainer = Trainer(train_label=train_data[0], train_sentence=cb_sent_rep_BOW_train, 
                         dev_label=dev_data[0], dev_sentence=cb_sent_rep_BOW_dev, 
                         test_label=test_data[0], test_sentence=cb_sent_rep_BOW_test, 
                         input_size=300, hidden_size=640, batch_size=128, learning_rate=7e-5, num_epochs=30)
    
    cb_bow_trainer.train()
    cb_bow_trainer.test()
    """
    
    
    gl_bow_trainer = Trainer(train_label=train_data[0], train_sentence=gl_sent_rep_BOW_train, 
                         dev_label=dev_data[0], dev_sentence=gl_sent_rep_BOW_dev, 
                         test_label=test_data[0], test_sentence=gl_sent_rep_BOW_test, 
                         input_size=300, hidden_size=720, batch_size=128, learning_rate=7e-3, num_epochs=30)
    
    gl_bow_trainer.train()
    gl_bow_trainer.test()
    

    """
    gl_bilstm_trainer = Trainer(train_label=train_data[0], train_sentence=gl_sent_rep_BiLSTM_train, 
                         dev_label=dev_data[0], dev_sentence=gl_sent_rep_BiLSTM_dev, 
                         test_label=test_data[0], test_sentence=gl_sent_rep_BiLSTM_test, 
                         input_size=300, hidden_size=720, batch_size=128, learning_rate=7e-5, num_epochs=30)
    
    gl_bilstm_trainer.train()
    gl_bilstm_trainer.test()
    """

if __name__ == "__main__":
    main()