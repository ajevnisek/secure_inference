import torch
import numpy as np

from research.communication.utils import Sender, Receiver
import time
from numba import njit, prange
num_bit_to_dtype = {
    8: np.ubyte,
    16: np.ushort,
    32: np.uintc,
    64: np.ulonglong
}

num_bit_to_sign_dtype = {
    32: np.int32,
    64: np.int64
}

num_bit_to_torch_dtype = {
    32: torch.int32,
    64: torch.int64
}


class Addresses:
    def __init__(self):
        self.port_01 = 12834
        self.port_10 = 12835
        self.port_02 = 12836
        self.port_20 = 12837
        self.port_12 = 12838
        self.port_21 = 12839

class NetworkAssets:
    def __init__(self, sender_01, sender_02, sender_12, receiver_01, receiver_02, receiver_12):
        # TODO: transfer only port
        self.receiver_12 = receiver_12
        self.receiver_02 = receiver_02
        self.receiver_01 = receiver_01
        self.sender_12 = sender_12
        self.sender_02 = sender_02
        self.sender_01 = sender_01

        if self.receiver_12:
            self.receiver_12.start()
        if self.receiver_02:
            self.receiver_02.start()
        if self.receiver_01:
            self.receiver_01.start()
        if self.sender_12:
            self.sender_12.start()
        if self.sender_02:
            self.sender_02.start()
        if self.sender_01:
            self.sender_01.start()


def get_assets(party):
    prf_01_seed = 0
    prf_02_seed = 1
    prf_12_seed = 2
    addresses = Addresses()

    if party == 0:

        crypto_assets = CryptoAssets(
            prf_01_numpy=np.random.default_rng(seed=prf_01_seed),
            prf_02_numpy=np.random.default_rng(seed=prf_02_seed),
            prf_12_numpy=None,
            prf_01_torch=torch.Generator().manual_seed(prf_01_seed),
            prf_02_torch=torch.Generator().manual_seed(prf_02_seed),
            prf_12_torch=None,
        )

        network_assets = NetworkAssets(
            sender_01=Sender(addresses.port_01),
            sender_02=Sender(addresses.port_02),
            sender_12=None,
            receiver_01=Receiver(addresses.port_10),
            receiver_02=Receiver(addresses.port_20),
            receiver_12=None
        )

    if party == 1:

        crypto_assets = CryptoAssets(
            prf_01_numpy=np.random.default_rng(seed=prf_01_seed),
            prf_02_numpy=None,
            prf_12_numpy=np.random.default_rng(seed=prf_12_seed),
            prf_01_torch=torch.Generator().manual_seed(prf_01_seed),
            prf_02_torch=None,
            prf_12_torch=torch.Generator().manual_seed(prf_12_seed),
        )

        network_assets = NetworkAssets(
            sender_01=Sender(addresses.port_10),
            sender_02=None,
            sender_12=Sender(addresses.port_12),
            receiver_01=Receiver(addresses.port_01),
            receiver_02=None,
            receiver_12=Receiver(addresses.port_21),
        )

    if party == 2:

        crypto_assets = CryptoAssets(
            prf_01_numpy=None,
            prf_02_numpy=np.random.default_rng(seed=prf_02_seed),
            prf_12_numpy=np.random.default_rng(seed=prf_12_seed),
            prf_01_torch=None,
            prf_02_torch=torch.Generator().manual_seed(prf_02_seed),
            prf_12_torch=torch.Generator().manual_seed(prf_12_seed),
        )

        network_assets = NetworkAssets(
            sender_01=None,
            sender_02=Sender(addresses.port_20),
            sender_12=Sender(addresses.port_21),
            receiver_01=None,
            receiver_02=Receiver(addresses.port_02),
            receiver_12=Receiver(addresses.port_12),
        )

    return crypto_assets, network_assets


NUM_BITS = 64
TRUNC = 10000
dtype = num_bit_to_dtype[NUM_BITS]
powers = np.arange(NUM_BITS, dtype=num_bit_to_dtype[NUM_BITS])[np.newaxis][:,::-1]
moduli = (2 ** powers)
P = 67

min_org_shit = -283206
max_org_shit = 287469
org_shit = (np.arange(min_org_shit, max_org_shit + 1) % P).astype(np.uint8)


def module_67(xxx):
    return org_shit[xxx.reshape(-1) - min_org_shit].reshape(xxx.shape)

#     orig_shape = list(value.shape)
#     value = value.reshape(-1, 1)
#     r_shift = value >> powers
#
#     value_bits = np.zeros(shape=(value.shape[0], 64), dtype=np.int8)
#     np.bitwise_and(r_shift, np.int8(1), out=value_bits)
#     return value_bits.reshape(orig_shape + [NUM_BITS])

def decompose(value, out=None, out_mask=None):
    orig_shape = list(value.shape)
    value = value.reshape(-1, 1)
    r_shift = value >> powers
    value_bits = np.zeros(shape=(value.shape[0], 64), dtype=np.int8)
    np.bitwise_and(r_shift, np.int8(1), out=value_bits)
    return value_bits.reshape(orig_shape + [NUM_BITS])

#

# def decompose(value):
#     orig_shape = list(value.shape)
#     value = value.reshape(-1, value.shape[3])
#     value_bits = decompose_(value)
#     return value_bits.reshape(orig_shape + [NUM_BITS])
#
# @njit(parallel=True)
# def decompose_(value):
#     out = np.zeros(shape=(value.shape[0], value.shape[1], 64), dtype=np.uint8)
#     for i in range(value.shape[0]):
#         for j in range(value.shape[1]):
#             out[i,j] = (value[i,j] & moduli) >> powers
#     return out
def sub_mode_p(x, y):
    mask = y > x
    ret = x - y
    ret_2 = x + (P - y)
    ret[mask] = ret_2[mask]
    return ret

class CryptoAssets:
    def __init__(self, prf_01_numpy, prf_02_numpy, prf_12_numpy, prf_01_torch, prf_02_torch, prf_12_torch):

        self.prf_12_torch = prf_12_torch
        self.prf_02_torch = prf_02_torch
        self.prf_01_torch = prf_01_torch
        self.prf_12_numpy = prf_12_numpy
        self.prf_02_numpy = prf_02_numpy
        self.prf_01_numpy = prf_01_numpy

        self.private_prf_numpy = np.random.default_rng(seed=31243)
        self.private_prf_torch = torch.Generator().manual_seed(31243)

        self.numpy_dtype = num_bit_to_dtype[NUM_BITS]
        self.torch_dtype = num_bit_to_torch_dtype[NUM_BITS]
        self.trunc = TRUNC

        self.numpy_max_val = np.iinfo(self.numpy_dtype).max
    def get_random_tensor_over_L(self, shape, prf):
        return torch.randint(
            low=torch.iinfo(self.torch_dtype).min // 2,
            high=torch.iinfo(self.torch_dtype).max // 2 + 1,
            size=shape,
            dtype=self.torch_dtype,
            generator=prf
        )



class SecureModule(torch.nn.Module):
    def __init__(self, crypto_assets, network_assets):

        super(SecureModule, self).__init__()

        self.crypto_assets = crypto_assets
        self.network_assets = network_assets

        self.trunc = TRUNC
        self.torch_dtype = num_bit_to_torch_dtype[NUM_BITS]
        self.dtype = num_bit_to_dtype[NUM_BITS]

        self.min_val = np.iinfo(self.dtype).min
        self.max_val = np.iinfo(self.dtype).max
        self.L_minus_1 = 2 ** NUM_BITS - 1
        self.signed_type = num_bit_to_sign_dtype[NUM_BITS]

    def add_mode_L_minus_one(self, a, b):
        ret = a + b
        ret[ret < a] += self.dtype(1)
        ret[ret == self.L_minus_1] = self.dtype(0)
        return ret

    def sub_mode_L_minus_one(self, a, b):
        ret = a - b
        ret[b > a] -= self.dtype(1)
        return ret



def fuse_conv_bn(conv_module, batch_norm_module):
    # TODO: this was copied from somewhere
    fusedconv = torch.nn.Conv2d(
        conv_module.in_channels,
        conv_module.out_channels,
        kernel_size=conv_module.kernel_size,
        stride=conv_module.stride,
        padding=conv_module.padding,
        bias=True
    )
    fusedconv.weight.requires_grad = False
    fusedconv.bias.requires_grad = False
    w_conv = conv_module.weight.clone().view(conv_module.out_channels, -1)
    w_bn = torch.diag(
        batch_norm_module.weight.div(torch.sqrt(batch_norm_module.eps + batch_norm_module.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
    if conv_module.bias is not None:
        b_conv = conv_module.bias
    else:
        b_conv = torch.zeros(conv_module.weight.size(0))
    b_bn = batch_norm_module.bias - batch_norm_module.weight.mul(batch_norm_module.running_mean).div(
        torch.sqrt(batch_norm_module.running_var + batch_norm_module.eps))
    fusedconv.bias.copy_(torch.matmul(w_bn, b_conv) + b_bn)

    W, B = fusedconv.weight, fusedconv.bias

    return W, B




def get_c(x_bits, multiplexer_bits, beta, j):
    t0 = time.time()
    beta = beta[..., np.newaxis]
    t1 = time.time()

    # multiplexer_bits = r_bits * (1 - beta) + t_bits * beta
    t2 = time.time()
    w = x_bits + j * multiplexer_bits - 2 * multiplexer_bits * x_bits
    t3 = time.time()
    w_cumsum = w.astype(np.int32)
    t4 = time.time()
    np.cumsum(w_cumsum, axis=-1, out=w_cumsum)
    np.subtract(w_cumsum, w, out=w_cumsum)
    rrr = w_cumsum
    # rrr = w.cumsum(axis=-1) - w
    t5 = time.time()
    zzz = j + (1 - 2 * beta) * (j * multiplexer_bits - x_bits)
    t6 = time.time()
    ret = rrr + zzz.astype(np.int32)
    t7 = time.time()
    # if j == 1:
    #     print("**************************************")
    #     print("get_c ", t1 - t0)
    #     print("get_c ", t2 - t1)
    #     print("get_c ", t3 - t2)
    #     print("get_c ", t4 - t3)
    #     print("get_c ", t5 - t4)
    #     print("get_c ", t6 - t5)
    #     print("get_c ", t7 - t6)
    #     print("**************************************")

    return ret

# def get_c_case_0(x_bits, r_bits, j):
#     x_bits = x_bits.astype(np.int32)
#     r_bits = r_bits.astype(np.int32)
#     j = j.astype(np.int32)
#
#     w = x_bits + j * r_bits - 2 * r_bits * x_bits
#     rrr = w[..., ::-1].cumsum(axis=-1)[..., ::-1] - w
#     zzz = j + j * r_bits  - x_bits
#     return ((rrr + zzz) % P).astype(np.uint64)
#
#
# def get_c_case_1(x_bits, t_bits, j):
#     x_bits = x_bits.astype(np.int32)
#     t_bits = t_bits.astype(np.int32)
#     j = j.astype(np.int32)
#
#     w = x_bits + j * t_bits - 2 * t_bits * x_bits
#     rrr = w[..., ::-1].cumsum(axis=-1)[..., ::-1] - w
#     zzz = j - j * t_bits + x_bits
#
#     return (zzz+rrr) % P

def get_c_case_2(u, j):
    c = (P + 1 - j) * (u + 1) + (P-j) * u
    c[..., 0] = u[...,0] * (P-1) ** j
    return c % P

import torch.nn as nn

class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W, _ = x.shape
        # print(N, C, H, W, self.block_size)
        x = x.reshape(N, C, H, W, self.block_size[0], self.block_size[1])
        x = x.transpose(0, 1, 2, 4, 3, 5)#.contiguous()
        x = x.reshape(N, C, H * self.block_size[0], W * self.block_size[1])
        return x


class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        # print(N, C, H, W, self.block_size)
        x = x.reshape(N, C, H // self.block_size[0], self.block_size[0], W // self.block_size[1], self.block_size[1])
        x = x.transpose(0, 1, 2, 4, 3, 5)#.contiguous()
        x = x.reshape(N, C, H // self.block_size[0], W // self.block_size[1], self.block_size[0] * self.block_size[1])
        return x




