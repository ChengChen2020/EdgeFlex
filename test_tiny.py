import argparse

from TinyImageNetDataLoader import TestTinyImageNetDataset, id_dict

import torch
import torch.utils.data
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testloader):
            if len(images) != args.bs:
                continue
            images, targets = images.to(device), targets.to(device)
            outputs, _ = net(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    print('Acc: %.3f%% (%d/%d)'
          % (100. * correct / total, correct, total))


parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Testing')
parser.add_argument('--bs', default=100, type=int, help='batch size')
parser.add_argument('--nu', default=5, type=int, help='number of devices')
parser.add_argument('--pp', default=5, type=int, help='partition point')
parser.add_argument('--n_embed', default=4096, type=int, help='codebook size')
parser.add_argument('--n_parts', default=1, type=int, help='number of parts')
args = parser.parse_args()

if __name__ == '__main__':
    transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))

    trainset = TrainTinyImageNetDataset(id=id_dict, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testset = TestTinyImageNetDataset(id=id_dict, transform=transform)
    testloader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=1)

    mobilenet_v2 = models.mobilenet_v2(num_classes=1000, width_mult=1.0).to(device)

    X = torch.rand(size=(1, 3, 64, 64)).to(device)

    for layer_idx, l in enumerate(mobilenet_v2.features):
        X = l(X)
        print(l.__class__.__name__, 'Output shape:\t', X.shape)

    mobilenet_v2.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'), strict=False)

    test(mobilenet_v2)
