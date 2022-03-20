import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# hyper parameters
epochs = 5
batch_size = 32
learning_rate = 1e-3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# prepare data
class DigitRecognizerDataset(Dataset):
    def __init__(self, root: str, train: bool = True) -> None:
        self.__data_dir = os.getcwd() + root + '/digit-recognizer'
        self.__train = train
        if train:
            self.__data_file = '/train.csv'
            self.__data = pd.read_csv(self.__data_dir + self.__data_file)
            self.__img_tensor = torch.reshape(torch.tensor(np.array(self.__data.iloc[:, 1:])),
                                              (len(self.__data), 1, 28, 28)).to(torch.float32)
            self.__label_tensor = torch.tensor(np.array(self.__data.iloc[:, 0]))
        else:
            self.__data_file = '/test.csv'
            self.__data = pd.read_csv(self.__data_dir + self.__data_file)
            self.__img_tensor = torch.reshape(torch.tensor(np.array(self.__data)),
                                              (len(self.__data), 1, 28, 28)).to(torch.float32)

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        img = self.__img_tensor[idx]
        if self.__train:
            label = self.__label_tensor[idx]
            return img, label
        else:
            return img


train_dataset = DigitRecognizerDataset(root='/data', train=True)
test_dataset = DigitRecognizerDataset(root='/data', train=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# describe neural network and set device
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.net(x)


model = LeNet().to(device)


# train func
def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)  # use model to predict
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            lossval, current = loss.item(), batch * len(X)
            print(f"loss: {lossval:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    model.eval()
    results = torch.tensor([], device=device, dtype=torch.int32)
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X).argmax(dim=1).to(torch.int32)
            results = torch.cat((results, pred), dim=0)
    return results


# test func
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epo in range(epochs):
    print(f"Epoch {epo + 1}\n-------------------------------")
    train(train_dataloader, model, loss_func, optimizer)
answer = test(test_dataloader, model)
answer_dataframe = pd.DataFrame(answer.to('cpu').numpy(), index=range(1, len(answer) + 1), columns=['Label'])
answer_dataframe.index.name = 'ImageId'
answer_dataframe.to_csv(os.getcwd() + '/data/digit-recognizer/submission.csv')
