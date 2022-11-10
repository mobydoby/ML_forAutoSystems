import math

def calcNextConvSize(in_dim, kernel_size, stride=(1,1), padding=0, dilation = (1,1)):
    z = in_dim[0]
    x_dim = math.floor((in_dim[1] + 2*padding - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
    y_dim = math.floor((in_dim[2] + 2*padding - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
    return z, x_dim, y_dim

def calcNextPoolSize(in_dim, kernel_size, stride, padding = 0):
    z = in_dim[0]
    x_out = math.floor((in_dim[1]+2*padding-(kernel_size[0]-1)-1)/stride[0] + 1)
    y_out = math.floor((in_dim[2]+2*padding-(kernel_size[1]-1)-1)/stride[1] + 1)
    return z, x_out, y_out

def calculate_fc_input(in_dim, con):
    layer_size = in_dim
    for i in range(len(conv)):
        layer_size = calcNextConvSize(layer_size, con[i]["kernel"], stride = con[i]["stride"], padding = con[i]["padding"])
        if pool_data[i] != None:
            layer_size = calcNextPoolSize(layer_size, pool_data[i]["kernel"], pool_data[i]["stride"])
        print(layer_size, con[i]["outputs"])
    return layer_size[1] * layer_size[2] * con[-1]["outputs"]

if __name__ == "__main__":

    conv= [
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
    pool_data = [
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

    IMAGE_HEIGHT = 3024
    IMAGE_WIDTH = 4032
    total_params = 0
    in_channels = 3
    rf1 = 8
    rf2 = 1.25
    in_dim = (3, int(IMAGE_HEIGHT / rf1 / rf2), int(IMAGE_WIDTH / rf1 / rf2))
    print(f"input size: {in_dim}")

    for i in range(len(conv)):
        # params per convolutional layer: (filter_w*filter_h*Input_Channels+1)*Output_Channels
        
        conv_layer_params = (conv[i]["kernel"][0]**2*in_channels+1)*conv[i]["outputs"]
        total_params += conv_layer_params
        if i!=0: in_channels = conv[i]["outputs"]
        print(f"conv layer {i} params: {conv_layer_params:>10}")
  
    print("---------------------------------")
    print("dimension calculations: ")
    fc_in = calculate_fc_input(in_dim, conv)
    print(f"fc input: {fc_in}")
    print("---------------------------------")


    fc = [fc_in, 2969, 1024, 512, 44, 44, 11]
    for i in range(len(fc)-1):
        layer_i_params = (fc[i]+1)*fc[i+1]
        print(f"fc layer {i} params: {layer_i_params:>12}")
        total_params+=layer_i_params

    print(f"total parmeters:     {total_params:,}")
