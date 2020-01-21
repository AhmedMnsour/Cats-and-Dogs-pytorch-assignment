import torch.nn as nn
import torch.nn.functional as F



# define the First CNN architecture
class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()

        # convolutional layer (sees 224x224x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3)
        
        # max pooling layer
        self.pool1 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 56x56x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3)

        # max pooling layer
        self.pool2 = nn.MaxPool2d(2, 2)
    
        # convolutional layer (sees 28x28x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3)

        # max pooling layer
        self.pool3 = nn.MaxPool2d(2, 2)

        
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)

        # linear layer (500 -> 2)
        self.fc2 = nn.Linear(128, 2)

        # dropout layer (p=0.25)
        self.dropout1 = nn.Dropout(0.2)

        # dropout layer (p=0.25)
        self.dropout2 = nn.Dropout(0.2)



    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = self.pool3(F.relu(self.conv3(x)))

        #x = self.pool4(F.relu(self.conv3(x)))
     
        # flatten image input
        x = x.view(-1, 64 * 14 * 14)

        # add dropout layer
        x = self.dropout1(x)

        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))

        # add dropout layer
        x = self.dropout2(x)

        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        
        return x



# define the Second CNN architecture
class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.fc1 = nn.Linear(32 * 53 * 53, 64)

        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x, verbose=False):
        # add sequence of convolutional and max pooling layers
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # flatten image input
        x = x.view(-1, 32*53*53)

        # add 1st hidden layer, with relu activation function
        x = self.fc1(x)
        x = F.relu(x)

        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x


