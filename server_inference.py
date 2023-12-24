import torch

import numpy as np
import socket

from arch import EnsembleNet

HOST = ''
PORT = 8888

dim = {3: 16, 5: 16, 8: 8}

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

while True:
    client_socket, addr = server_socket.accept()

    try:
        data = client_socket.recv(4096)
        binary_string = data.decode('utf-8')
        # print(binary_string)
        pp = int(binary_string[0:12], 2)
        n_embed = int(binary_string[12:24], 2)
        n_parts = int(binary_string[24:36], 2)

        print(pp, n_embed, n_parts)

        net = EnsembleNet(res_stop=pp, ncls=100, skip_quant=False, n_embed=n_embed, n_parts=n_parts).cuda()
        exp_name = f'0.0001_{pp}_100_5_{n_embed}_{n_parts}_1.0_False_AdaptE'
        exp_path = f'./checkpoint/ckpt_{exp_name}/'
        checkpoint = torch.load(f'{exp_path}/1.pth')
        checkpoint_2 = torch.load(f'{exp_path}/-1.pth')
        net.quantizer.load_state_dict(checkpoint_2['quantizer'])
        net.decoder.load_state_dict(checkpoint['decoder'])

        array_size = dim[pp] * dim[pp] * n_parts * 12 + 36
        received_array = np.array([int(binary_string[i:i+12], 2) for i in range(36, array_size, 12)], dtype=np.int64).reshape((1, dim[pp], dim[pp], n_parts))

        indices = torch.from_numpy(received_array).cuda()
        X = net.quantizer.get_codes_from_indices(indices)
        #print(X.shape)
        X = X.view((X.shape[0], X.shape[3], X.shape[1], X.shape[2]))
        client_socket.sendall(net.decoder(X).detach().cpu().numpy().tobytes())

    except KeyboardInterrupt:
        break

    finally:
        client_socket.close()
