import torch
from torch.utils.data import Dataset
import pandas as pd
import lightning as L
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

class LinearModel(L.LightningModule):
    def __init__(self, input_size: int, n_classes: int, hidden_size: list, **params):
        super().__init__()
        # params validation
        assert("lr" in params)
        assert("weight_decay" in params)
        self.lr = params['lr']
        self.weight_decay = params['weight_decay']

        # model creation
        layers = []
        for i in range(len(hidden_size)):
            if i == 0:
                layers.append(Linear(input_size, hidden_size[i]))
            else:
                layers.append(Linear(hidden_size[i-1], hidden_size[i]))
        layers.append(Linear(hidden_size[-1], n_classes))
        self.model = torch.nn.Sequential(*layers)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class LendingClubDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_dataloader(features, labels, batch_size):
    dataset = LendingClubDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size)

def hyperparameter_tuning(model: torch.nn.Module,
                          random=False):
    if random: # random search
        pass
    else: # grid search
        pass

def train(model: torch.nn.Module,
          feature: torch.tensor,
          label: torch.tensor,
          n_epochs: int = 50,
          batch_size: int = 32,
          device: str = 'auto',
          validation_split: float = 0.2):
    # split dataset
    train_X, val_X, train_y, val_y = train_test_split(
        feature, label, test_size=validation_split, stratify=label)
    # convert to torch tensor
    train_loader = get_dataloader(train_X, train_y, batch_size)
    val_loader = get_dataloader(val_X, val_y, batch_size)
    # train model
    trainer = L.Trainer(max_epochs=n_epochs, devices=device)
    trainer.fit(model, train_loader, val_dataloaders=val_loader)


def unit_test():
    # load dataset
    #dataset = pd.read_csv('data/loan.csv')
    # get 100 samples
    train_dataset = torch.rand(100, 784)
    train_label = torch.randint(0, 10, (100,))
    # train model
    model_param = {'lr': 1e-3, 'weight_decay': 1e-5}
    model = LinearModel(784, 10, [128, 64], **model_param)
    train(model, train_dataset, train_label, n_epochs=50, batch_size=32, device='auto', validation_split=0.2)

if __name__ == '__main__':
    unit_test()