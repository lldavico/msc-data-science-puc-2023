import torch
import torchvision
import math
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3* 32 * 32, 480)
        self.fc2 = nn.Linear(480, 240)
        self.fc3 = nn.Linear(240, 120)
        self.fc4 = nn.Linear(120, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Vgg(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.features = make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def LoadCIFAR10(batch_size, use_data_augmentation, donwload=True):
    if use_data_augmentation:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, 4),
                            transforms.ToTensor(),
                            normalize,
                        ])
        test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
                        ])
    else:
        train_transform = transforms.Compose( 
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = train_transform


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=donwload, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=donwload, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader


def GetPrecision(net, dataloader, device, criterion):
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels) 
    return loss, 100 * correct / total



# Hyper-Parámetros 
batch_size = 64
weight_decay = 0.0001   # Factor para la regularización L2
learning_rate = 0.0001
use_data_augmentation = True

# Acá hay que descomentar la red que se quiera usar
net = MLP()
# net = SimpleCNN()
# net = Vgg([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']) # VGG-11
# net = Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']) # VGG-16

# Cargamos el set de entrenamiento y test
trainloader, testloader = LoadCIFAR10(batch_size, use_data_augmentation)

# Usamos GPU si existe alguna
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

net.to(device)
# Elegimos la función de pérdida y el optimizador.
# ADAM incluye internamente un regularizador L2 y su factor es el 'weight_decay'
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Comenzamos el loop de entrenamiento
for epoch in range(1000): 

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # Torch nos fuerza a indicarle que queremos borrar los gradientes guardados
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    current_epoch = epoch + 1

    if (current_epoch % 10) == 0:
        # Mostramos estadísticas del aprendizaje (incluyendo rendimiento en train y test)

        train_loss, train_performance = GetPrecision(net, trainloader, device, criterion)
        test_loss, test_performance = GetPrecision(net, testloader, device, criterion)
        
        # Nomalizo las funciones de pérdida para que sean comparables entre train y test
        # Ocurre que el set de train es 5 veces más grande que el set de test... por eso las multiplico por 5
        train_loss *= batch_size
        test_loss *= batch_size * 5

        print(f'Ep. {current_epoch}. Train loss: {train_loss :.3f}. Test loss: {test_loss :.3f}. Train acc: {train_performance :.2f}. Test acc: {test_performance :.2f}')


print('Finished Training')
