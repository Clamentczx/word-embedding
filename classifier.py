import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Define the neural network model
class SentenceClassifier(nn.Module):
    """
    This is a classifier network with one hidden layer.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(SentenceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

class ToTensor:
    """
    This class is used to transfer labels to one-hot tensors.
    """
    def __init__(self, train_label, dev_label, test_label):
        self.train_label = train_label
        self.dev_label = dev_label
        self.test_label = test_label

    def get_all_label(self):
        """
        Returns all posible labels.
        """
        return list(set(self.train_label + self.dev_label + self.test_label))
    
    def get_total_label(self):
        """
        Returns the number of labels.
        """
        return len(self.get_all_label())

    def label_to_one_hot(self):
            """
            Returns all the one-hot label tensors.
            """
            all_label = self.get_all_label()
            print("There are {} labels in total.".format(len(all_label)))
            
            train_label_tensor = torch.nn.functional.one_hot(torch.tensor([all_label.index(l) for l in self.train_label]), num_classes=self.get_total_label()).to(torch.float)
            dev_label_tensor = torch.nn.functional.one_hot(torch.tensor([all_label.index(l) for l in self.dev_label]), num_classes=self.get_total_label()).to(torch.float)
            test_label_tensor = torch.nn.functional.one_hot(torch.tensor([all_label.index(l) for l in self.test_label]), num_classes=self.get_total_label()).to(torch.float)

            return train_label_tensor, dev_label_tensor, test_label_tensor
    

class ToDataset(Dataset):
    """
    This is a class that is for creating a database for Dataloader.
    """
    def __init__(self, dataA, dataB):
        self.dataA = dataA
        self.dataB = dataB
    
    def __len__(self):
        return len(self.dataA)
    
    def __getitem__(self, idx):
        x = self.dataA[idx].clone().detach()
        y = self.dataB[idx].clone().detach()
        return x, y

class Trainer:
    """
    A trainer class.
    """
    def __init__(self, train_label, train_sentence, dev_label, dev_sentence, 
                 test_label, test_sentence, input_size=300, hidden_size=50, 
                 batch_size=128, learning_rate=0.01, num_epochs=10):
        
        # load data
        t = ToTensor(train_label, dev_label, test_label)
        self.train_label_tensor, self.dev_label_tensor, self.test_label_tensor = t.label_to_one_hot()

        self.batch_size = batch_size

        # define the model, loss function, and optimizer
        self.model = SentenceClassifier(input_size, hidden_size, t.get_total_label())
        self.loss_f = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # move model to gpu if available
        self.model.to(self.device)
        self.train_sentence_tensor = torch.stack(train_sentence).to(self.device)
        self.dev_sentence_tensor = torch.stack(dev_sentence).to(self.device)
        self.test_sentence_tensor = torch.stack(test_sentence).to(self.device)


    def train(self):
        # create dataset
        train_dataset = ToDataset(self.train_sentence_tensor, self.train_label_tensor.to(self.device))
        dev_dataset = ToDataset(self.dev_sentence_tensor, self.dev_label_tensor.to(self.device))

        # create dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)
        # train

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for batch_data, batch_labels in train_dataloader:
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_data)

                loss = self.loss_f(predictions, batch_labels)

                # l1 norm
                l1_lambda = 0.001
                l1_norm = sum(torch.linalg.norm(p, 1) for p in self.model.parameters())
                loss = loss + l1_lambda * l1_norm

                loss.backward()

                self.optimizer.step()

                train_loss += loss
            train_loss /= len(train_dataloader)

            # Evaluate the model on the development set
            self.model.eval()
            dev_loss = 0
            dev_correct = 0
            dev_count = 0
            for batch_data, batch_labels in dev_dataloader:
                predictions = self.model(batch_data)

                loss = self.loss_f(predictions, batch_labels)

                dev_correct += (torch.argmax(predictions, 1) == torch.argmax(batch_labels, 1)).sum().item()
                dev_count += batch_labels.size(0)

                dev_loss += loss
                    
            dev_loss /= len(dev_dataloader)
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Dev Loss={dev_loss:.4f}, Dev Acc={dev_correct/dev_count:.4f}")

    def test(self):
        # create test dataset and dataloader
        test_dataset = ToDataset(self.test_sentence_tensor, self.test_label_tensor.to(self.device))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # evaluate on test set
        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            test_correct = 0
            test_count = 0
            for batch_data, batch_labels in test_dataloader:
                predictions = self.model(batch_data)

                loss = self.loss_f(predictions, batch_labels)

                test_loss += loss
                test_correct += (torch.argmax(predictions, 1) == torch.argmax(batch_labels, 1)).sum().item()
                test_count += batch_labels.size(0)

            test_loss /= len(test_dataloader)
            print(f"Test Loss={test_loss:.4f}, Test Acc={test_correct/test_count:.4f}")