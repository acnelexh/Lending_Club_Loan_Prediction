from typing import Any
import torch
from torch.utils.data import Dataset
import pandas as pd
import lightning as L
from torchmetrics.functional import accuracy, precision, recall, f1_score as f1
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

        self.save_hyperparameters()

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
        self.log("train_loss", loss)
        self.log("train_acc", accuracy(y_hat, y, task="multiclass", num_classes=2))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("validation_loss", loss)
        self.log("validation_acc", accuracy(y_hat, y, task="multiclass", num_classes=2))
        self.log("validation_f1", f1(y_hat, y, task="multiclass", num_classes=2))
        self.log("validation_precision", precision(y_hat, y, task="multiclass", num_classes=2))
        self.log("validation_recall", recall(y_hat, y, task="multiclass", num_classes=2))

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", accuracy(y_hat, y, task="multiclass", num_classes=2))
        self.log("test_f1", f1(y_hat, y, task="multiclass", num_classes=2))
        self.log("test_precision", precision(y_hat, y, task="multiclass", num_classes=2))
        self.log("test_recall", recall(y_hat, y, task="multiclass", num_classes=2))

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        return super().predict_step(*args, **kwargs)

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

def hyperparameter_tuning(model_fn: torch.nn.Module,
                          val_X: torch.tensor,
                          val_y: torch.tensor,
                          random=False,
                          random_search_itr=100,
                          max_epochs=10,
                          metric='f1'):
    # lr, weight_decay, batch_size, hidden_size
    trainer = L.Trainer(max_epochs=max_epochs, devices='auto')
    result = {}
    if random: # random search
        for i in range(random_search_itr):
            # create model
            lr = 10 ** (-5 * torch.rand(1)).item()
            weight_decay = 10 ** (-5 * torch.rand(1)) .item()
            batch_size = (2 ** torch.randint(1, 8, (1,))).item() # 2 - 258
            model = model_fn(10, 2, [32, 32], lr=lr, weight_decay=weight_decay)
            val_loader = get_dataloader(val_X, val_y, batch_size)
            # train and evalute on validation set
            trainer.fit(model, val_loader)
            if metric == 'f1':
                result[(lr, weight_decay, batch_size)] = trainer.validate(model, val_loader)[0]['validation_f1']
            elif metric == 'acc':
                result[(lr, weight_decay, batch_size)] = trainer.validate(model, val_loader)[0]['validation_acc']
            elif metric == 'precision':
                result[(lr, weight_decay, batch_size)] = trainer.validate(model, val_loader)[0]['validation_precision']
            elif metric == 'recall':
                result[(lr, weight_decay, batch_size)] = trainer.validate(model, val_loader)[0]['validation_recall']
    # return the best hyperparameters on metrics
    best_param = max(result, key=result.get)
    best_param = {'lr': best_param[0], 'weight_decay': best_param[1], 'batch_size': best_param[2]}
    return best_param
        

def pipeline(model: torch.nn.Module,
             train_X: torch.tensor,
             train_y: torch.tensor,
             test_X: torch.tensor,
             test_y: torch.tensor,
             n_epochs: int = 50,
             batch_size: int = 32,
             device: str = 'auto',
             validation_split: float = 0.2):
    # split dataset and setup loader
    train_X, val_X, train_y, val_y = train_test_split(
        train_X, train_y, test_size=validation_split, stratify=train_y)
    train_loader = get_dataloader(train_X, train_y, batch_size)
    val_loader = get_dataloader(val_X, val_y, batch_size)
    test_loader = get_dataloader(test_X, test_y, batch_size)
    # set up trainer
    trainer = L.Trainer(max_epochs=n_epochs, devices=device)
    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    # train
    trainer.test(model, test_loader)

def unit_test_pipeline():
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    X_test = torch.randn(100, 10)
    y_test = torch.randint(0, 2, (100,))
    model = LinearModel(10, 2, [32, 32], lr=0.01, weight_decay=0.001)
    pipeline(model, X_train, y_train, X_test, y_test)

def unit_test_hyperparameter_tuning():
    X_val = torch.randn(100, 10)
    y_val = torch.randint(0, 2, (100,))
    hyperparameter_tuning(LinearModel, X_val, y_val, random=True, random_search_itr=5)

if __name__ == "__main__":
    #unit_test_pipeline()
    unit_test_hyperparameter_tuning()