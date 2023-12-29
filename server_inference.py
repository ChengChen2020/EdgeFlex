import torch

import numpy as np
import socket

from arch import EnsembleNet

HOST = ''
PORT = 8888

dim = {3: 16, 5: 16, 8: 8}

# 16 * 16 * 8 * 12 / 8

# 3KB

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

pp, n_embed, n_parts = 5, 4096, 8

net = EnsembleNet(res_stop=pp, ncls=100, skip_quant=False, n_embed=n_embed, n_parts=n_parts).cuda()
exp_name = f'0.0001_{pp}_100_5_{n_embed}_{n_parts}_1.0_False_AdaptE'
exp_path = f'./checkpoint/ckpt_{exp_name}/'
checkpoint = torch.load(f'{exp_path}/1.pth')
print(checkpoint['acc'])
checkpoint_2 = torch.load(f'{exp_path}/-1.pth')
net.quantizer.load_state_dict(checkpoint_2['quantizer'])
net.decoder.load_state_dict(checkpoint['decoder'])

client_socket, addr = server_socket.accept()

while True:

    try:
        data = client_socket.recv(4096)
        print(len(data))

        if not data:
            break

        binary_string = ''.join([format(int(data[i]), f'08b') for i in range(len(data))])
        # print(binary_string)
        pp = int(binary_string[0:16], 2)
        n_embed = int(binary_string[16:32], 2)
        n_parts = int(binary_string[32:48], 2)

        # print(pp, n_embed, n_parts)

        index_length = int(np.log2(n_embed))

        array_size = dim[pp] * dim[pp] * n_parts * index_length + 48
        received_array = np.array([int(binary_string[i:i+index_length], 2) for i in range(48, array_size, index_length)], dtype=np.int64).reshape((1, dim[pp], dim[pp], n_parts))

        indices = torch.from_numpy(received_array).cuda()

        print(indices.shape)
        X = net.quantizer.get_codes_from_indices(indices)
        print(X.shape)

        X = X.view((X.shape[0], X.shape[3], X.shape[1], X.shape[2]))

        results = net.decoder(X).detach().cpu().numpy()
        print(results)
        # print(len(results.tobytes()))
        client_socket.sendall(results.tobytes())

    except KeyboardInterrupt:
        break

    # finally:
    #     client_socket.close()

client_socket.close()
