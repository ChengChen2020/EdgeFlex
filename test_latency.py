import torch
import torch.utils.data

import torchvision
import torchvision.transforms as transforms

import time
import argparse
import numpy as np

from arch import EnsembleNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Testing')
parser.add_argument('--nu', default=5, type=int, help='number of devices')
parser.add_argument('--pp', default=5, type=int, help='partition point')
parser.add_argument('--n_embed', default=4096, type=int, help='codebook size')
parser.add_argument('--n_parts', default=8, type=int, help='number of parts')
args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(testset, np.arange(0, 2000)), batch_size=1, shuffle=False, num_workers=1)
# testloader = torch.utils.data.DataLoader(
#        testset, batch_size=100, shuffle=False, num_workers=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    print('Acc: %.3f%% (%d/%d)'
          % (100. * correct / total, correct, total))


def ensemble_test(pp=5, nu=5, n_embed=2048, n_parts=2):
    net = EnsembleNet(res_stop=pp, ncls=100, skip_quant=False, n_embed=n_embed, n_parts=n_parts).to(device)

    exp_name = f'0.0001_{pp}_100_{nu}_{n_embed}_{n_parts}_1.0_False_AdaptE'
    exp_path = f'./checkpoint/ckpt_{exp_name}/'

    X = torch.rand(size=(2, 3, 32, 32)).to(device)
    print(net(X)[0].shape)

    entries = len(testloader)
    print(entries)

    enc_time = []
    quant_time = []
    dec_time = []

    for num_of_ens in range(nu):

        checkpoint = torch.load(f'{exp_path}/{num_of_ens}.pth')
        checkpoint_2 = torch.load(f'{exp_path}/-1.pth')
        print(checkpoint['acc'])
        net.encoder.load_state_dict(checkpoint_2['encoder'])
        net.quantizer.load_state_dict(checkpoint_2['quantizer'])
        net.decoder.load_state_dict(checkpoint['decoder'])

        test(net)
        net.eval()

        print(num_of_ens)

        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(testloader):
                X_test, y_test = X_test.to(device), y_test.to(device)
                start_time = time.time()

                X = net.encoder(X_test)
                enc_time.append(time.time() - start_time)

                X = X.view((X.shape[0], X.shape[2], X.shape[3], X.shape[1]))
                X, indices, commit_loss = net.quantize(X)
                quant_time.append(time.time() - start_time - enc_time[-1])

                X = X.view((X.shape[0], X.shape[3], X.shape[1], X.shape[2]))
                X = net.decoder(X)
                dec_time.append(time.time() - start_time - enc_time[-1] - quant_time[-1])

        print(len(enc_time), "{:.3f}".format(sum(enc_time[1:]) / (len(enc_time) - 1) * 1000.))
        print(len(quant_time), "{:.3f}".format(sum(quant_time[1:]) / (len(quant_time) - 1) * 1000.))
        print(len(dec_time), "{:.3f}".format(sum(dec_time[1:]) / (len(dec_time) - 1) * 1000.))


if __name__ == "__main__":
    ensemble_test(nu=args.nu, pp=args.pp, n_embed=args.n_embed, n_parts=args.n_parts)
