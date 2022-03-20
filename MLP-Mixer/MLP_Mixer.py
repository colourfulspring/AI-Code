import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

# 禁止import除了torch以外的其他包，依赖这几个包已经可以完成实验了

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, data):
        return torch.transpose(data, 0, 1)


class Mixer_Layer(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super(Mixer_Layer, self).__init__()
        ########################################################################
        # 这里需要写Mixer_Layer（layernorm，mlp1，mlp2，skip_connection）
        self.patch_mixer = nn.Sequential(
            nn.LayerNorm([patch_size ** 2, (28 // patch_size) ** 2]),
            Transpose(),
            nn.Linear((28 // patch_size) ** 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, (28 // patch_size) ** 2),
            Transpose()
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm([patch_size ** 2, (28 // patch_size) ** 2]),
            nn.Linear((28 // patch_size) ** 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, (28 // patch_size) ** 2),
        )

        ########################################################################

    def forward(self, x):
        x1 = x + self.patch_mixer(x)
        y = x1 + self.channel_mixer(x1)
        return y

    ########################################################################

    ########################################################################


class MLPMixer(nn.Module):
    def __init__(self, patch_size, hidden_dim, depth):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        ########################################################################
        # 这里写Pre-patch Fully-connected, Global average pooling, fully connected
        self.net = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size),
            *([Mixer_Layer(patch_size, hidden_dim)] * depth),
            nn.Fold(kernel_size=patch_size, stride=patch_size, output_size=28),
            nn.AvgPool2d(kernel_size=patch_size),
            nn.Flatten(),
            nn.Linear((28 // patch_size) ** 2, 10),
        )
        ########################################################################

    def forward(self, data):
        return self.net(data)

    ########################################################################

    # 注意维度的变化

    ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 计算loss并进行优化
            model.zero_grad()
            answer = model(data)
            loss = criterion(answer, target)
            loss.backward()
            optimizer.step()
            ########################################################################
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.
    num_correct = 0  # correct的个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 需要计算测试集的loss和accuracy
            answer = model(data)
            num_correct += torch.eq(answer.argmax(1), target).sum()
            test_loss += criterion(answer, target)
        accuracy = num_correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        ########################################################################
        print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(test_loss, accuracy))


if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 2e-2

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)

    ########################################################################
    model = MLPMixer(patch_size=4, hidden_dim=16, depth=2).to(device)  # 参数自己设定，其中depth必须大于1
    # 这里需要调用optimizer，criterion(交叉熵)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    ########################################################################

    train(model, train_loader, optimizer, n_epochs, criterion)
    test(model, test_loader, criterion)
