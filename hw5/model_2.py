from multiprocessing import pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FC(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))
            

        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))
        
    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)

class CNN(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim   


        self.c_data = [
            {
                # conv 1 
                "outputs": 128,
                "kernel": (17,17),
                "stride": (6,6),
                "padding": 0
            },
            {
                # conv 2
                "outputs": 256,
                "kernel": (7,7),
                "stride": (1,1),
                "padding": 1
            },
            {
                # conv 3
                "outputs": 384,
                "kernel": (5,5),
                "stride": (1,1),
                "padding": 1
            },
            {
                # conv 4
                "outputs": 384,
                "kernel": (3,3),
                "stride": (1,1),
                "padding": 1
            },
            {
                # conv 4
                "outputs": 256,
                "kernel": (3,3),
                "stride": (1,1),
                "padding": 1
            }
        ]

        self.pool_data = [
            {
                "kernel": (3,3),
                "stride": (2,2)
            },
            {
                "kernel": (3,3),
                "stride": (2,2)
            },
            None,
            None,
            {
                "kernel": (3,3),
                "stride": (2,2)
            },
        ]

        self.conv_out_dim = self.calculate_fc_input()

        self.conv_stack = nn.Sequential(
            #conv layer 1
            nn.Conv2d(self.in_dim[0], self.c_data[0]["outputs"], self.c_data[0]["kernel"], self.c_data[0]["stride"], self.c_data[0]["padding"]),
            nn.ReLU(),
            nn.BatchNorm2d(self.c_data[0]["outputs"]),
            nn.MaxPool2d(self.pool_data[0]["kernel"], self.pool_data[0]["stride"]),
            
            #conv layer 2
            nn.Conv2d(self.c_data[0]["outputs"], self.c_data[1]["outputs"], self.c_data[1]["kernel"], self.c_data[1]["stride"], self.c_data[1]["padding"]),
            nn.BatchNorm2d(self.c_data[1]["outputs"]),        
            nn.ReLU(),

            nn.MaxPool2d(self.pool_data[1]["kernel"], self.pool_data[1]["stride"]),
            
            # #conv layer 3
            nn.Conv2d(self.c_data[1]["outputs"], self.c_data[2]["outputs"], self.c_data[2]["kernel"], self.c_data[2]["stride"], self.c_data[2]["padding"]),
            nn.BatchNorm2d(self.c_data[2]["outputs"]),
            nn.ReLU(),
            # nn.MaxPool2d(self.pool_data[2]["kernel"], self.pool_data[2]["stride"]),    

            nn.Conv2d(self.c_data[2]["outputs"], self.c_data[3]["outputs"], self.c_data[3]["kernel"], self.c_data[3]["stride"], self.c_data[3]["padding"]),
            nn.BatchNorm2d(self.c_data[3]["outputs"]),            
            nn.ReLU(),
            # nn.MaxPool2d(self.pool_data[3]["kernel"], self.pool_data[3]["stride"]),    
  
            nn.Conv2d(self.c_data[3]["outputs"], self.c_data[4]["outputs"], self.c_data[4]["kernel"], self.c_data[4]["stride"], self.c_data[4]["padding"]),
            nn.ReLU(),
            nn.BatchNorm2d(self.c_data[4]["outputs"]),
            nn.MaxPool2d(self.pool_data[4]["kernel"], self.pool_data[4]["stride"]),
        )     

        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),

            nn.Linear(self.conv_out_dim, 2969),
            nn.ReLU(),
            nn.BatchNorm1d(2969),

            # nn.Dropout(p=0.5),
            nn.Linear(2969, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),

            nn.Linear(512, 44),
            nn.ReLU(),
            nn.BatchNorm1d(44),

            nn.Linear(44, 44),
            nn.ReLU(),
            nn.BatchNorm1d(44),

            nn.Linear(44, 11),
            nn.LogSoftmax(dim=1)
        )

    def _calcNextConvSize(self, in_dim, kernel_size, stride=(1,1), padding=0, dilation = (1,1)):
        z = in_dim[0]
        x_dim = math.floor((in_dim[1] + 2*padding - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
        y_dim = math.floor((in_dim[2] + 2*padding - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
        return z, x_dim, y_dim
    
    def _calcNextPoolSize(self, in_dim, kernel_size, stride, padding = 0):
        z = in_dim[0]
        x_out = math.floor((in_dim[1]+2*padding-(kernel_size[0]-1)-1)/stride[0] + 1)
        y_out = math.floor((in_dim[2]+2*padding-(kernel_size[1]-1)-1)/stride[1] + 1)
        return z, x_out, y_out

    def calculate_fc_input(self):
        layer_size = self.in_dim
        for i in range(len(self.c_data)):
            layer_size = self._calcNextConvSize(layer_size, self.c_data[i]["kernel"], stride = self.c_data[i]["stride"], padding = self.c_data[i]["padding"])
            if self.pool_data[i] != None:
                layer_size = self._calcNextPoolSize(layer_size, self.pool_data[i]["kernel"], self.pool_data[i]["stride"])
        print(f"Conv Output Dimensions: {self.c_data[-1]['outputs']}, {layer_size[1]}, {layer_size[2]}")
        return layer_size[1] * layer_size[2] * self.c_data[-1]["outputs"]

    # def test_calc_fc(self):

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

class CNN_small(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # [(W−K+2P)/S]+1
        self.in_dim = in_dim # 3 x 32 x 32
        self.out_dim = out_dim # 10

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride = 1, padding = 2), #output is 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2,2), 
            # nn.Conv2d(12, 16, (4, 4), stride = 1, padding = 0), # output is 12 x 25 x 25
            # nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride = 1, padding = 1), # output is 16 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 32 x 8 x 8
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(32*25*25, 1000),
            # nn.ReLU(),
            nn.Linear(32*8*8, 120),
            nn.ReLU(),
            nn.Linear(120, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x
        
