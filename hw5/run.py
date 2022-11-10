import torch
import torch.nn.functional as F
from model_2 import CNN
import sys
import torchvision.transforms as T
from dataset_new import BuildingDataset

IMAGE_HEIGHT = 3024
IMAGE_WIDTH = 4032

def test(net, loader, device):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:

            data, target = data.to(device), target.to(device)
            
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
            
            total = total + 1

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))), flush=True)
    
    return 100.0 * correct / len(loader.dataset)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Incorrect Usage: <python run.py [cnn_model_path] [data_path] [label_path]>")
        exit(1)

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    rf1 = 8
    rf2 = 1.25

    final_input_h = int(IMAGE_HEIGHT / rf1 / rf2)
    final_input_w = int(IMAGE_WIDTH / rf1 / rf2)

    input_dim = (3, final_input_h, final_input_w)
    out_dim = 11

    network = CNN(in_dim=input_dim, out_dim=out_dim)
    network = network.to(device)

    #load model from 
    MODEL_PATH = sys.argv[1]
    DATA_PATH = sys.argv[2]
    LABEL_PATH = sys.argv[3]

    network.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    resize_final = T.Resize(size = (final_input_h, final_input_w))
    convert = T.ConvertImageDtype(torch.float)
    normalize = T.Normalize(mean=[0.46063736, 0.47591286, 0.46565274], std=[0.225, 0.225, 0.225])

    test_transforms = T.Compose([resize_final, convert, normalize])  
    test_dataset = BuildingDataset(LABEL_PATH, DATA_PATH, transform_pre=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, len(test_dataset), shuffle=False)

    test(network, test_loader, device)
