from torch import nn

class Diehl_2015(nn.Module):
    def __init__(self, bias=False):
        super(Diehl_2015, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=0, bias=bias)
        self.avrpool1_1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=5, stride=1, padding=0, bias=bias)
        self.avrpool2_1 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(in_features=1024, out_features=10, bias=bias)
        self.drop = nn.Dropout2d(p=0.5)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.avrpool1_1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.avrpool2_1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return output



class SNN_Huynh2021(nn.Module):
    def __init__(self, batch_size: int):
        super(SNN_Huynh2021, self).__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(5, 5), stride=(1, 1), padding=0, bias=False)
        self.conv1_if = CW_softIF(batch_size=batch_size,num_channels=12,height=24, width=24,reset=0.0)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=0,
                               bias=False)
        self.conv2_if = CW_softIF(batch_size=batch_size, num_channels=64, height=8, width=8,reset=0.0)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(in_features=1024, out_features=10, bias=False)
        self.fc1_if = PB_neurons(batch_size=batch_size,num_features=10)

        self.flat = nn.Flatten()

        self.store = []


    def init_neuron_models(self):
        for m in self.named_children():
            if isinstance(m[1], PB_neurons) or isinstance(m[1], CW_softIF):
                m[1].init_vars()

    def update_threshold(self, threshold: List[float]) -> None:
        self.conv1_if.update_threshold(threshold[0])
        self.conv2_if.update_threshold(threshold[1])
        # self.fc1_if.threshold = threshold[2]

    def forward(self, images: torch.Tensor, encode = False, ):

        x= images
        if encode:
            rand_list = torch.rand(images.size()).cuda()
            x = torch.zeros(images.size()).cuda()
            x.masked_fill_(rand_list < images, 1.)

        x = self.conv1(x)
        x = self.conv1_if(x)
        self.store.append(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv2_if(x)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc1_if(x)
        return x
