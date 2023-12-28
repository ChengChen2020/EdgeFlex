import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from torchvision.models.quantization import QuantizableMobileNetV2, MobileNet_V2_QuantizedWeights

# from utils import progress_bar
from vector_quantize_pytorch import VectorQuantize

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MobileNetXX(nn.Module):
    def __init__(self, encdec, n_embed=1024, n_parts=2, decay=0.8, commitment=1.0):
        super().__init__()
        self.encoder = encdec['encoder']
        self.quant_dim = self.encoder(torch.zeros(1, 3, 32, 32)).shape[1]
        self.decoder = encdec['decoder']
        self.classifier = encdec['classifier']
        self.n_embed = n_embed
        self.n_parts = n_parts
        self.decay = decay
        self.commitment = commitment
        self.quantizer = VectorQuantize(dim=self.quant_dim // self.n_parts,
                                        codebook_size=self.n_embed,  # size of the dictionary
                                        decay=self.decay,
                                        # the exponential moving average decay, lower means the
                                        # dictionary will change faster
                                        commitment_weight=self.commitment)

    def quantize(self, z_e):
        # Split along the specified dimension
        z_e_split = torch.split(z_e, self.quant_dim // self.n_parts, dim=3)

        z_q_parts, indices_parts, commit_loss = [], [], 0

        for z_e_part in z_e_split:
            a, b, c, d = z_e_part.shape

            # Use view for reshaping
            z_e_part_reshaped = z_e_part.view(a, -1, d)

            # Quantize the part
            z_q_part, indices_part, commit_loss_part = self.quantizer(z_e_part_reshaped)

            # Update commit loss
            commit_loss += commit_loss_part

            # Reshape back to the original shape
            z_q_part_reshaped = z_q_part.view(a, b, c, d)
            indices_part_reshaped = indices_part.view(a, b, c)

            # Append to the lists
            z_q_parts.append(z_q_part_reshaped)
            indices_parts.append(indices_part_reshaped)

        # Concatenate along the specified dimension
        z_q = torch.cat(z_q_parts, dim=3)
        indices = torch.stack(indices_parts, dim=3)

        return z_q, indices, commit_loss

    def forward(self, X):
        # 2, 3, 32, 32
        X = self.encoder(X)

        # 2, 64, 8, 8
        X = X.view((X.shape[0], X.shape[2], X.shape[3], X.shape[1]))
        X, indices, commit_loss = self.quantize(X)

        # offload = indices.detach().cpu().numpy()
        # print(offload.shape)

        # X = self.quantizer.get_codes_from_indices(indices)
        X = X.view((X.shape[0], X.shape[3], X.shape[1], X.shape[2]))

        X = self.decoder(X)

        X = nn.functional.adaptive_avg_pool2d(X, 1).reshape(X.shape[0], -1)

        return self.classifier(X), commit_loss


def get_model(pp):
    mobilenet_v2 = models.mobilenet_v2(num_classes=1000, width_mult=1.0)

    mobilenet_v2.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'), strict=True)

    encoder_layers = []
    decoder_layers = []
    classifier_layers = []

    # X = torch.rand(size=(2, 3, 64, 64))

    for layer_idx, l in enumerate(mobilenet_v2.features):
        # X = l(X)
        # print(l.__class__.__name__, 'Output shape:\t', X.shape)
        if layer_idx <= pp:
            encoder_layers.append(l)
        else:
            decoder_layers.append(l)

    for layer_idx, l in enumerate(mobilenet_v2.classifier):
        # print(l.__class__.__name__)
        classifier_layers.append(l)

    # for layer_idx, l in enumerate(mobilenet_v2.named_children()):
    #     print(l.__class__.__name__)

    return dict(encoder=nn.Sequential(*encoder_layers),
                decoder=nn.Sequential(*decoder_layers),
                classifier=nn.Sequential(*classifier_layers))


def train_quantizer(pp, nb, np, lr, ep, trainloader, testloader):
    print(f'Device: {torch.cuda.get_device_name(0)}')

    net = nn.DataParallel(MobileNetXX(get_model(pp=pp), n_embed=nb, n_parts=np))
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    for param in net.module.encoder.parameters():
        param.requires_grad = False
    net.module.encoder.eval()
    # for param in net.module.decoder.parameters():
    #     param.requires_grad = False
    # net.module.decoder.eval()

    best_acc = 0
    acc_list = []

    for epoch in range(0, ep):

        print('\nEpoch: %d' % epoch)
        net.module.quantizer.train()
        net.module.decoder.train()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (images, targets) in enumerate(trainloader):
            images, targets = images.to(device), targets.to(device)
            outputs, commit_loss = net(images)
            loss = criterion(outputs, targets) + commit_loss
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            train_loss += loss.mean().item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / len(trainloader), 100. * correct / total, correct, total))

        scheduler.step()

        net.module.quantizer.eval()
        net.module.decoder.eval()

        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(testloader):
                images, targets = images.to(device), targets.to(device)
                outputs, commit_loss = net(images)
                loss = criterion(outputs, targets) + commit_loss
                test_loss += loss.mean().item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss / len(testloader), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'encoder': net.module.encoder.state_dict(),
                'quantizer': net.module.quantizer.state_dict(),
                'decoder': net.module.decoder.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, f'Config_{pp}_{nb}_{np}_{lr}.pth')
            best_acc = acc

        acc_list.append(acc)


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--train_bs', default=64, type=int)
parser.add_argument('--test_bs', default=100, type=int)
parser.add_argument('--pp', default=5, type=int, help='partition point')
parser.add_argument('--ep', default=100, type=int, help='epochs')
parser.add_argument('--nu', default=5, type=int, help='num of users')
parser.add_argument('--n_embed', default=4096, type=int, help='embedding size')
parser.add_argument('--n_parts', default=8, type=int, help='number of parts')
parser.add_argument('--commitment', default=1.0, type=int, help='commitment')
parser.add_argument('--skip_quant', action='store_true', help='skip quantization')
parser.add_argument('--quant', default='VQ', help='number of parts')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

if __name__ == '__main__':
    # mobilenet_v2 = models.mobilenet_v2(num_classes=1000, width_mult=1.0).to(device)
    # print(device)
    # mobilenet_v2 = models.quantization.mobilenet_v2(weights=weights, quantize=True).to(device)
    # print(len(testset))
    # print(weights.meta['categories'])
    # for X, y in testloader:
    #     print(X.shape, y.shape)
    #     break
    #
    # for X, y in trainloader:
    #     print(X.shape, y.shape)
    #     break
    #
    # print(len(trainset))
    # print(len(trainloader))

    # mobilenet_v2.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'), strict=True)

    # test(mobilenet_v2)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.ImageNet(
        root='./data', split='train', transform=transform_train)

    testset = torchvision.datasets.ImageNet(
        root='./data', split='val', transform=transform_test)

    # print(len(trainset))

    train_quantizer(args.pp, args.n_embed, args.n_parts, args.lr, args.ep,
                    DataLoader(trainset, batch_size=256, shuffle=True, num_workers=32),
                    DataLoader(testset, batch_size=256, shuffle=False, num_workers=32))
