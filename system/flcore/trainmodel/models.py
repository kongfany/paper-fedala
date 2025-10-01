import torch
import torch.nn as nn


batch_size = 16

class LocalModel(nn.Module):
    def __init__(self, feature_extractor, head):
        super(LocalModel, self).__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        
    def forward(self, x, feat=False):
        out = self.feature_extractor(x)
        if feat:
            return out
        else:
            out = self.head(out)
            return out


class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024, dim1=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim1), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(dim1, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = z

        return out

if __name__ == '__main__':

    x =torch.randn(10, 1, 28, 28)
    net =FedAvgCNN()
    #print(net)

    # x1 = net(x)
    # print(x1.shape)
    x1 =net.conv1(x)
    x2 =net.conv2(x1)
    x3 =torch.flatten(x2, 1)
    x4 = net.fc1(x3)
    x5 = net.fc(x4)
    print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
    print(x5)
