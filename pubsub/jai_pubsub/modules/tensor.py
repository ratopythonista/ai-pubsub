import math

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from jai_pubsub.modules import JaiTask
from jai_pubsub.modules.networks import MyNet


class Tensor(JaiTask):
    def __init__(self, learning_rate: int) -> None:
        self.learning_rate = learning_rate

    def compile(self):
        dtype = torch.float
        device = torch.device("cpu")
        # device = torch.device("cuda:0") # Uncomment this to run on GPU

        # Create random input and output data
        self.x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
        self.y = torch.sin(self.x)

        # Randomly initialize weights
        self.a = torch.randn((), device=device, dtype=dtype)
        self.b = torch.randn((), device=device, dtype=dtype)
        self.c = torch.randn((), device=device, dtype=dtype)
        self.d = torch.randn((), device=device, dtype=dtype)

        return self

    def train(self):
        for t in range(2000):
            # Forward pass: compute predicted y
            y_pred = (
                self.a + self.b * self.x + self.c * self.x**2 + self.d * self.x**3
            )

            # Compute and print loss
            loss = (y_pred - self.y).pow(2).sum().item()
            if t % 100 == 99:
                print(t, loss)

            # Backprop to compute gradients of a, b, c, d with respect to loss
            grad_y_pred = 2.0 * (y_pred - self.y)
            grad_a = grad_y_pred.sum()
            grad_b = (grad_y_pred * self.x).sum()
            grad_c = (grad_y_pred * self.x**2).sum()
            grad_d = (grad_y_pred * self.x**3).sum()

            # Update weights using gradient descent
            self.a -= self.learning_rate * grad_a
            self.b -= self.learning_rate * grad_b
            self.c -= self.learning_rate * grad_c
            self.d -= self.learning_rate * grad_d

    def run(self):

        return f"Result: y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3"


class TensonMultiprocess(JaiTask):
    def __init__(self, batch_size: int) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )


        self.batch_size = batch_size

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def compile(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.my_net = MyNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.my_net.parameters(), lr=0.001, momentum=0.9)
        self.my_net.to(self.device)
        return self

    def run(self):
        for data in self.trainloader:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            self.criterion(self.my_net(inputs), labels).backward()
            self.optimizer.step()  # This will update the shared parameters
