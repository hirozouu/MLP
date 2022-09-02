# module
from yaml import load
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# const
MY_BATCH_SIZE = 100

# 手書き数字
class HandwrittenDigit:
    data_train = None
    data_test = None
    train_loader = None
    test_loader = None

    def __init__(self) -> None:    
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.data_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.data_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(self.data_train,batch_size=MY_BATCH_SIZE,shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.data_test,batch_size=MY_BATCH_SIZE,shuffle=False)


def main():
    data = HandwrittenDigit()
    for i in range(28):
        for j in range(28):
            print(f'{data.data_train.data[0][i][j]:4}', end='')
    print()
    plt.imshow(data.data_train.data[0], cmap='gray')
    plt.show()
    print(data.data_train.targets[0].item())

if __name__ == '__main__':
    main()

