import torch
import torch.utils.data
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.models as models
# import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# import os
import time
import queue
import argparse
import threading
import numpy as np

from arch import EnsembleNet

ckpt_path = "checkpoint_1_8"

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Testing')
parser.add_argument('--bs', default=100, type=int, help='batch size')
parser.add_argument('--nu', default=5, type=int, help='number of devices')
parser.add_argument('--pp', default=5, type=int, help='partition point')
parser.add_argument('--n_embed', default=4096, type=int, help='codebook size')
parser.add_argument('--n_parts', default=1, type=int, help='number of parts')
parser.add_argument('--off', action='store_true', help='offloading')
args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=args.bs, shuffle=False, num_workers=1)
testloader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(testset, np.arange(0, 10000)), batch_size=args.bs, shuffle=False, num_workers=10)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if len(inputs) != args.bs:
                continue
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    print('Acc: %.3f%% (%d/%d)'
          % (100. * correct / total, correct, total))


def ensemble_test(bs=100, nu=5, pp=5, n_embed=4096, n_parts=2):
    net = EnsembleNet(res_stop=pp, ncls=100, skip_quant=False, n_embed=n_embed, n_parts=n_parts).to(device)

    exp_name = f'0.0001_{pp}_100_{nu}_{n_embed}_{n_parts}_1.0_False_AdaptE'
    exp_path = f'./checkpoint/ckpt_{exp_name}/'

    X = torch.rand(size=(2, 3, 32, 32)).to(device)

    print(net(X)[0].shape)

    if args.off:
        num_users = 2
    else:
        num_users = 2
    batch_size = bs
    entries = len(testloader)
    print(entries)
    y_hat_tensor = torch.empty([num_users, entries, batch_size, 100])
    accuracy_ensemble_tensor = torch.empty([num_users])

    def accuracy(y, ensemble_y_pred):
        ens_pred = torch.max(ensemble_y_pred.data, 1)[1]
        return (ens_pred == y).sum()

    # for num_of_ens in range(num_users):
    for num_of_ens in range(1):

        checkpoint = torch.load(f'{exp_path}/{num_of_ens}.pth')
        checkpoint_2 = torch.load(f'{exp_path}/-1.pth')
        print(checkpoint['acc'])
        # print(checkpoint['epoch'])
        net.encoder.load_state_dict(checkpoint_2['encoder'])
        net.quantizer.load_state_dict(checkpoint_2['quantizer'])
        print(net.quantizer.codebook.shape)
        print(net.quantizer.codebook.ndim)
        net.decoder.load_state_dict(checkpoint['decoder'])
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']

        # print("????", test(net))

        net.eval()
        # print(num_of_ens)

        # Create Socket for Offloading
        if args.off:
            import socket

            HOST = '128.46.74.214'
            PORT = 8888

            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))

        # Begin Evaluating
        with torch.no_grad():

            time_list = []
            result_queue = queue.Queue()

            for b, (X_test, y_test) in enumerate(testloader):
                X_test, y_test = X_test.to(device), y_test.to(device)

                if args.off:

                    start_time = time.time()

                    binary_strings = [format(a, '016b') for a in [pp, n_embed, n_parts]]
                    X, indices = net.offload_forward(X_test)
                    indices = indices.detach().cpu().numpy()
                    index_length = int(np.log2(n_embed))

                    print(b, type(indices), indices.shape, type(index_length))
                    binary_strings.extend([format(a, f'0{index_length}b') for a in indices.flatten()])
                    binary_strings = ''.join(binary_strings)

                    # length = len(binary_strings)

                    # print(length, binary_strings)

                    # Pad the binary string to make its length a multiple of 8
                    # binary_strings = binary_strings + '0' * (8 - len(binary_strings) % 8)

                    # Convert binary string to binary data
                    binary_data = bytes([int(binary_strings[i:i + 8], 2) for i in range(0, len(binary_strings), 8)])
                    # print(binary_data)

                    # print(len(binary_data), binary_data)

                    # binary_strings = [format(int(binary_data[i]), f'08b') for i in range(len(binary_data) - 1)]
                    # bs = ''.join(binary_strings)
                    # print(bs, len(bs))

                    collab = threading.Thread(target=offloading, args=(client_socket, binary_data, result_queue))
                    collab.start()

                    local_results = net.decoder(X)

                    y_hat_tensor[num_of_ens, b, :, :] = local_results

                    collab.join()

                    offload_results = result_queue.get()
                    # print(offload_results)

                    y_hat_tensor[1, b, :, :] = torch.from_numpy(offload_results)

                    ct = time.time() - start_time
                    time_list.append(ct)

                else:
                    # results = net(X_test)[0]
                    print(X_test.shape)
                    _, indices = net.offload_forward(X_test)

                    binary_strings = [format(a, '016b') for a in [pp, n_embed, n_parts]]
                    X, indices = net.offload_forward(X_test)
                    indices = indices.detach().cpu().numpy()
                    index_length = int(np.log2(n_embed))

                    print(b, type(indices), indices.shape, type(index_length))
                    binary_strings.extend([format(a, f'0{index_length}b') for a in indices.flatten()])
                    binary_strings = ''.join(binary_strings)

                    binary_data = bytes([int(binary_strings[i:i + 8], 2) for i in range(0, len(binary_strings), 8)])

                    # OFFLOADING

                    binary_string = ''.join([format(int(binary_data[i]), f'08b') for i in range(len(binary_data))])
                    # print(binary_string)
                    pp = int(binary_string[0:16], 2)
                    n_embed = int(binary_string[16:32], 2)
                    n_parts = int(binary_string[32:48], 2)

                    # print(pp, n_embed, n_parts)

                    index_length = int(np.log2(n_embed))

                    array_size = 16 * 16 * n_parts * index_length + 48
                    received_array = np.array(
                        [int(binary_string[i:i + index_length], 2) for i in range(48, array_size, index_length)],
                        dtype=np.int64).reshape((1, 16, 16, n_parts))

                    indices = torch.from_numpy(received_array).cuda()

                    # from einops import rearrange
                    # print(net.quantizer.codebook.shape)
                    # X = rearrange(net.quantizer.codebook[torch.from_numpy(indices)], '... h d -> ... (h d)')
                    X = net.quantizer.get_codes_from_indices(indices)
                    X = X.view((X.shape[0], X.shape[3], X.shape[1], X.shape[2]))
                    results = net.decoder(X)
                    # print(results)
                    y_hat_tensor[num_of_ens, b, :, :] = results

        if args.off:
            client_socket.close()

    print("Ensemble...")

    new_testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)

    # 5 * 100 * 100 * 100
    # 5 * 200 * 50 * 100
    ensemble_y_hat = torch.empty([num_users, 100, 100, 100])
    ensemble_accuracy_per_users = torch.empty([num_users, 100])

    for b, (X_test, y_test) in enumerate(new_testloader):
        for num_of_ens in range(num_users):
            preds = y_hat_tensor[:num_of_ens + 1, :, :, :]
            ensemble_y_hat[num_of_ens, :, :, :] = (
                torch.mean(preds.view([num_of_ens + 1, -1]), dim=0).view([-1, 100, 100]))
            y_pred = ensemble_y_hat[num_of_ens, b, :, :]
            # print(y_pred.shape)
            batch_ens_corr = accuracy(y_test, y_pred)
            ensemble_accuracy_per_users[num_of_ens, b] = batch_ens_corr

    for num_of_ens in range(num_users):
        total_correct = ensemble_accuracy_per_users[num_of_ens, :]
        sum_of_correct = total_correct.sum()
        acc_correct = sum_of_correct / 100.
        accuracy_ensemble_tensor[num_of_ens] = acc_correct
    #
    for i in range(num_users):
        print(f'Accuracy of Ensemble of {i + 1} Models: {accuracy_ensemble_tensor[i].item():.3f}%')

    if args.off:
        print(np.mean(time_list))


def offloading(client_socket, data, out_queue):
    client_socket.sendall(data)
    acknowledgement = client_socket.recv(4096)
    result = np.frombuffer(acknowledgement, dtype=np.float32).reshape(1, 100)
    out_queue.put(result)


if __name__ == "__main__":
    ensemble_test(bs=args.bs, nu=args.nu, pp=args.pp, n_embed=args.n_embed, n_parts=args.n_parts)
