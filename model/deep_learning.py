import torch
from torch.utils.data import Dataset
import lightning as L
from torchmetrics.functional import accuracy, precision, recall, f1_score as f1
from torch.nn import Linear
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class LinearModel(L.LightningModule):
    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 hidden_size: list,
                 lr: float,
                 weight_decay: float, **kwargs):
        super().__init__()
        # params validation
        self.lr = lr
        self.weight_decay = weight_decay

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
        self.log("train_acc", accuracy(y_hat, y, task="multiclass", num_classes=2, average="macro"))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("validation_loss", loss)
        self.log("validation_acc", accuracy(y_hat, y, task="multiclass", num_classes=2, average="macro"))
        self.log("validation_f1", f1(y_hat, y, task="multiclass", num_classes=2, average="macro"))
        self.log("validation_precision", precision(y_hat, y, task="multiclass", num_classes=2, average="macro"))
        self.log("validation_recall", recall(y_hat, y, task="multiclass", num_classes=2, average="macro"))

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", accuracy(y_hat, y, task="multiclass", num_classes=2, average="macro"))
        self.log("test_f1", f1(y_hat, y, task="multiclass", num_classes=2, average="macro"))
        self.log("test_precision", precision(y_hat, y, task="multiclass", num_classes=2, average="macro"))
        self.log("test_recall", recall(y_hat, y, task="multiclass", num_classes=2, average="macro"))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class LinearModelBatchNorm(LinearModel):
    # same as LinearModel but with batch normalization
    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 hidden_size: list,
                 lr: float,
                 weight_decay: float, **kwargs):
        super().__init__(input_size, n_classes, hidden_size, lr, weight_decay)
        # add batch normalization and replace linear layers
        layers = []
        for i in range(len(hidden_size)):
            if i == 0:
                layers.append(torch.nn.Sequential(Linear(input_size, hidden_size[i]), torch.nn.BatchNorm1d(hidden_size[i])))
            else:
                layers.append(torch.nn.Sequential(Linear(hidden_size[i-1], hidden_size[i]), torch.nn.BatchNorm1d(hidden_size[i])))
        layers.append(Linear(hidden_size[-1], n_classes))
        self.model = torch.nn.Sequential(*layers)

class LinearModelDropout(LinearModel):
    # same as LinearModel but with dropout
    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 hidden_size: list,
                 lr: float,
                 weight_decay: float, **kwargs):
        super().__init__(input_size, n_classes, hidden_size, lr, weight_decay)
        # add batch normalization and replace linear layers
        layers = []
        for i in range(len(hidden_size)):
            if i == 0:
                layers.append(torch.nn.Sequential(Linear(input_size, hidden_size[i]), torch.nn.Dropout(0.2)))
            else:
                layers.append(torch.nn.Sequential(Linear(hidden_size[i-1], hidden_size[i]), torch.nn.Dropout(0.2)))
        layers.append(Linear(hidden_size[-1], n_classes))
        self.model = torch.nn.Sequential(*layers)

class LinearModelBatchNormDropout(LinearModel):
    # same as LinearModel but with batch normalization and dropout
    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 hidden_size: list,
                 drop_out: float,
                 lr: float,
                 weight_decay: float, **kwargs):
        super().__init__(input_size, n_classes, hidden_size, lr, weight_decay)
        # add batch normalization and replace linear layers
        layers = []
        for i in range(len(hidden_size)):
            if i == 0:
                layers.append(torch.nn.Sequential(Linear(input_size, hidden_size[i]),
                                                  torch.nn.BatchNorm1d(hidden_size[i]),
                                                  torch.nn.Dropout(drop_out)))
            else:
                layers.append(torch.nn.Sequential(Linear(hidden_size[i-1], hidden_size[i]),
                                                  torch.nn.BatchNorm1d(hidden_size[i]),
                                                  torch.nn.Dropout(drop_out)))
        layers.append(Linear(hidden_size[-1], n_classes))
        self.model = torch.nn.Sequential(*layers)

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
                          model_params: dict,
                          val_X: torch.tensor,
                          val_y: torch.tensor,
                          random_search_itr=100,
                          max_epochs=10,
                          metric='f1'):
    # lr, weight_decay, batch_size, hidden_size
    result = {}
    for i in range(random_search_itr):
        trainer = L.Trainer(max_epochs=max_epochs, devices='auto', )
        # create model
        lr = 10 ** (-5 * torch.rand(1)).item()
        weight_decay = 10 ** (-5 * torch.rand(1)).item()
        batch_size = (2 ** torch.randint(4, 7, (1,))).item() # 16 - 128
        model = model_fn(**model_params, lr=lr, weight_decay=weight_decay)
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
        

def pipeline(model_fn: torch.nn.Module,
             model_params: dict,
             train_X: torch.tensor,
             train_y: torch.tensor,
             test_X: torch.tensor,
             test_y: torch.tensor,
             n_epochs: int = 50,
             device: str = 'auto',
             validation_split: float = 0.2,
             random_search_itr: int = 100,
             hyperparam_metric: str = 'f1',
             hyperparam_epoch: int = 10):
    # split dataset and setup loader
    train_X, val_X, train_y, val_y = train_test_split(
        train_X, train_y, test_size=validation_split, stratify=train_y)
    # hyperparameter tuning
    hyperparam = hyperparameter_tuning(model_fn,
                                       model_params,
                                       val_X,
                                       val_y,
                                       random_search_itr,
                                       hyperparam_epoch,
                                       hyperparam_metric)
    # setup dataloader
    train_loader = get_dataloader(train_X, train_y, hyperparam['batch_size'])
    val_loader = get_dataloader(val_X, val_y, hyperparam['batch_size'])
    test_loader = get_dataloader(test_X, test_y, hyperparam['batch_size'])
    # set up trainer
    trainer = L.Trainer(max_epochs=n_epochs, devices=device)
    model = model_fn(**model_params, **hyperparam)
    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    # train
    trainer.test(model, test_loader)

def unit_test_pipeline():
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    X_test = torch.randn(100, 10)
    y_test = torch.randint(0, 2, (100,))
    model_params = {"input_size": X_train.shape[1], "n_classes": 2,"hidden_size": [128, 128]}
    pipeline(
        LinearModel,
        model_params,
        train_X=X_train,
        train_y=y_train,
        test_X=X_test,
        test_y= y_test,
        n_epochs=5,
        device='auto',
        validation_split=0.2,
        random_search_itr=10,
        hyperparam_metric='f1',
        hyperparam_epoch=10)

def unit_test_models():
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    X_test = torch.randn(100, 10)
    y_test = torch.randint(0, 2, (100,))
    model_params = {"input_size": X_train.shape[1], "n_classes": 2,"hidden_size": [128, 128], 'drop_out': 0.2}
    pipeline(
        LinearModelBatchNormDropout,
        model_params,
        train_X=X_train,
        train_y=y_train,
        test_X=X_test,
        test_y= y_test,
        n_epochs=5,
        device='auto',
        validation_split=0.2,
        random_search_itr=10,
        hyperparam_metric='f1',
        hyperparam_epoch=10)


if __name__ == "__main__":
    #unit_test_pipeline()
    #unit_test_pipeline()
    unit_test_models()
