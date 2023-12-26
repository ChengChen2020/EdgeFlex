import torch
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.quantization import QuantizableMobileNetV2, MobileNet_V2_QuantizedWeights

from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def test(net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testloader):
            images, targets = images.to(device), targets.to(device)
            outputs = net(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                         % (100. * correct / total, correct, total))

    print('Acc: %.3f%% (%d/%d)'
          % (100. * correct / total, correct, total))


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


# trainset = torchvision.datasets.ImageNet(
#     root='./data', split='train', transform=transform_train)

testset = torchvision.datasets.ImageNet(
    root='./data', split='val', transform=transform_test)

testloader = DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=1)

weights = MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1

if __name__ == '__main__':

    # mobilenet_v2 = models.mobilenet_v2(num_classes=1000, width_mult=1.0).to(device)
    print(device)
    mobilenet_v2 = models.quantization.mobilenet_v2(weights=weights, quantize=True).to(device)
    print(len(testset))
    # print(weights.meta['categories'])
    # for X, y in testloader:
    #     print(X.shape, y.shape)

    # mobilenet_v2.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'), strict=True)

    test(mobilenet_v2)
