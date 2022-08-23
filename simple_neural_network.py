import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np 

# set seed
seed = 888
torch.manual_seed(seed)

# dataset
mnist_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((1.0,), (0.5,))])
root = './data' # if not exist, download mnist dataset
mnist_train_set = datasets.MNIST(root=root, train=True, transform=mnist_trans, download=True)
mnist_test_set = datasets.MNIST(root=root, train=False, transform=mnist_trans, download=True)

# data loaders
batch_size = 600

mnist_train_loader = torch.utils.data.DataLoader(
                 dataset=mnist_train_set,
                 batch_size=batch_size,
                 shuffle=True)
mnist_test_loader = torch.utils.data.DataLoader(
                dataset=mnist_test_set,
                batch_size=batch_size,
                shuffle=False)           

# simlpe neural network
layer_1_neurons = 64
layer_2_neurons = 64
layer_3_neurons = 64
layer_4_neurons = 10

class Neural_network(nn.Module):
    def __init__(self):
        # super calls the method (__init__() in this case) of the parent class (nn.Module in this case) and passing an object to it automatically (self in this case). This is equivalent to super(Neural_network, self).__init__() in python 3
        # https://realpython.com/python-super/ 
        # https://stackoverflow.com/questions/61288224/why-not-super-init-model-self-in-pytorch
        super().__init__()
        
        self.fc1 = nn.Linear(28*28*1,layer_1_neurons)
        self.fc2 = nn.Linear(layer_1_neurons,layer_2_neurons)
        self.fc3 = nn.Linear(layer_2_neurons,layer_3_neurons)
        self.fc4 = nn.Linear(layer_3_neurons,layer_4_neurons)

    def forward(self, x):
        x = x.view(-1, 28*28*1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x


simple_neural_network = Neural_network().cuda()

# optimizer
simple_neural_network_lr = 1e-3 # <<<<<----------- change learning rate here

opt = optim.SGD(simple_neural_network.parameters(), lr = simple_neural_network_lr)

# loss function
criterion = nn.CrossEntropyLoss()

# list to hold losses
test_losses = []
test_accuracies = []

# training loop
epochs = 10
batch = 600

for i in range(epochs):
    for batch_idx, (x, target) in enumerate(mnist_train_loader):
        x = x.cuda()
        target = target.cuda()

        out = simple_neural_network(x)
        loss = criterion(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f'epoch: {i+1}, loss: {loss.item()}')

    with torch.no_grad():
        test_loss = []
        test_accuracy = []
            
        for batch_idx, (x, target) in enumerate(mnist_test_loader):
            x, target = x.cuda(), target.cuda()
            outputs = simple_neural_network(x)
            _, predicted = torch.max(outputs.data, 1)
            batch_test_loss = criterion(outputs, target)
            test_loss.append(batch_test_loss.item())
            test_accuracy.append((predicted == target).sum().item() / predicted.size(0))
        
        test_accuracies.append(np.mean(test_accuracy))

# save final test losses
# torch.save(test_losses, f'test_losses_lr_{simple_neural_network_lr}.pt')
torch.save(test_accuracies, f'test_accuracies_lr_{simple_neural_network_lr}.pt')
