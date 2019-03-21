

import os
import inflect
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from dataset import ImageDataset
from segmentation import find

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, 5, 1)
        self.conv2 = nn.Conv2d(25, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50,10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 25 * 24 *24
        x = F.max_pool2d(x, 2, 2) # 25 * 12 * 12
        x = F.relu(self.conv2(x)) # 50 * 8 * 8
        x = F.max_pool2d(x, 2, 2) # 50 * 4 * 4
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

root = './data'
def predict(model,data):

    for i, inputs in enumerate(loader):


            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)



            return int(preds)


#change dowload to True
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=False)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=False)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

img_input = input('Enter Image path: \n')
digit_images = find(img_input)
s=''
for i in digit_images.keys():

    loader = ImageDataset(digit_images[i])

    loader = val = torch.utils.data.DataLoader(loader,batch_size=1,
                                               shuffle=True)


    model = MNIST()
    model.load_state_dict(torch.load('MNIST.pth'))
    model.eval()
    s+=str(predict(model,loader))
    os.remove(digit_images[i])
s = int(s)
conv = inflect.engine()
print(s , conv.number_to_words(s))
