from torch import nn

class Diehl_2015(nn.Module):
    def __init__(self, bias=False):
        super(Diehl_2015, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=0, bias=bias)
        self.avrpool1_1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=5, stride=1, padding=0, bias=bias)
        self.avrpool2_1 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(in_features=1024, out_features=10, bias=bias)
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(x)
        x = self.drop(x)
        x = self.avrpool1_1(x)
        x = self.conv2(x)
        x = nn.ReLU(x)
        x = self.drop(x)
        x = self.avrpool2_1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
