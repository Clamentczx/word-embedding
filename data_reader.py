import os
import torch

class Data_reader:
    """
    A class that reads row data from txt files.
    """
    def __init__(self, path):
        self.path = path
        self.train = []
        self.dev = []
        self.test = []
        self.stopwords = []

    # read data
    def read(self):
        """
        A data reader. 
        """
        with open(os.path.join(self.path, 'train.txt')) as f:
            for row in f:
                self.train.append(row.strip())

        with open(os.path.join(self.path,'dev.txt')) as f:
            for row in f:
                self.dev.append(row.strip())

        with open(os.path.join(self.path,'test.txt')) as f:
            for row in f:
                self.test.append(row.strip())

        # read stopwords 
        with open('stopwords.txt') as f:
            for row in f:
                self.stopwords.append(row.strip())
        # a stopword remover
    def remove_stopword(self, list, stopwords):
            return [word for word in list if word not in set(stopwords)]

    # split data to coarse, fine and non-label data
    def split_data(self, list):
        """
        A data spliter that seperate coarse, fine, and text list.
        Additionally, stopwords are removed from here.
        """
        coarse_list = []
        fine_list = []
        text_list = []

        for item in list:
            fine = item.split(" ", 1)[0]
            coarse = fine.split(":", 1)[0]

            text = item.split(" ", 1)[1]

            coarse_list.append(coarse)
            fine_list.append(fine)

            words = self.remove_stopword(text.lower().split(" "), self.stopwords)
            text_list.append(" ".join(words))
            
        return coarse_list, fine_list, text_list
    
    def get_data(self):
        """
        A getter for read data.
        """
        self.read()
        train_data = self.split_data(self.train)
        dev_data = self.split_data(self.dev)
        test_data = self.split_data(self.test)

        return train_data, dev_data, test_data
    
    @staticmethod
    def writer(path, weight, file_name):
        """
        Saves the weight to a text file
        """
        with open(os.path.join(path, "{}.txt".format(file_name)), "w") as f:
            for i in range(weight.shape[0]):
                vector = " ".join([str(x) for x in weight[i].tolist()])
                f.write(f"{vector}\n")
    
    @staticmethod
    def loader(path, file_name):
        """
        Loads the weight to a text file
        """
        loaded_word_embeddings = torch.zeros(7991, 300)
        with open(os.path.join(path, "{}.txt".format(file_name)), "r") as f:
            for i, line in enumerate(f):
                line = line.strip().split()
                vector = [float(x) for x in line]
                loaded_word_embeddings[i] = torch.tensor(vector)
        return loaded_word_embeddings


if __name__ == "__main__":
    dr = Data_reader('data')
    print(dr.get_data())

