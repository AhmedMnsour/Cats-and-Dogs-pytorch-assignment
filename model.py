import torch.nn as nn
import torch.nn.functional as F



# define the First CNN architecture
class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()

        # convolutional layer (sees 224x224x3 image tensor)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 56x56x16 tensor)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
    
        # convolutional layer (sees 28x28x32 tensor)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)

        # convolutional layer (sees 28x28x32 tensor)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(16 * 14 * 14, 500)

        # linear layer (500 -> 2)
        self.fc2 = nn.Linear(500, 2)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.pool(F.relu(self.conv4(x)))
     
        # flatten image input
        x = x.view(-1, 16 * 14 * 14)

        # add dropout layer
        x = self.dropout(x)

        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))

        # add dropout layer
        x = self.dropout(x)

        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        
        return x



# define the Second CNN architecture
class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 53 * 53, 50)

        self.fc2 = nn.Linear(50, 2)
        
    def forward(self, x, verbose=False):
        # add sequence of convolutional and max pooling layers
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # flatten image input
        x = x.view(-1, 16*53*53)

        # add 1st hidden layer, with relu activation function
        x = self.fc1(x)
        x = F.relu(x)

        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x


