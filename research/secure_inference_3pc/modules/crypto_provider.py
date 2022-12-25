import numpy as np
import torch

from research.secure_inference_3pc.modules.base import PRFFetcherModule, SecureModule
from research.secure_inference_3pc.modules.conv2d import get_output_shape
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, P, MIN_VAL, MAX_VAL, SIGNED_DTYPE, NUM_BITS, NUM_OF_COMPARE_BITS
from research.secure_inference_3pc.timer import Timer
from research.secure_inference_3pc.base import decompose, get_c, module_67, TypeConverter, SpaceToDepth

# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W_shape, stride, dilation, padding, groups, crypto_assets, network_assets):
        super(PRFFetcherConv2D, self).__init__(crypto_assets, network_assets)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):

        out_shape = get_output_shape(X_share, self.W_shape, self.padding, self.dilation, self.stride)

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=SIGNED_DTYPE)

        return np.zeros(shape=out_shape, dtype=X_share.dtype)


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherPrivateCompare, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_0):
        return 


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherShareConvert, self).__init__(crypto_assets, network_assets)
        self.private_compare = PRFFetcherPrivateCompare(crypto_assets, network_assets)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(dummy_tensor.shape) + [NUM_OF_COMPARE_BITS], dtype=np.int8)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val, size=dummy_tensor.shape, dtype=self.dtype)

        self.private_compare(dummy_tensor)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val, size=dummy_tensor.shape, dtype=self.dtype)

        return


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherMultiplication, self).__init__(crypto_assets, network_assets)

    def forward(self, dummy_tensor):

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        return dummy_tensor


class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherMSB, self).__init__(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)
        self.private_compare = PRFFetcherPrivateCompare(crypto_assets, network_assets)

    def forward(self, dummy_tensor):
        size = dummy_tensor.shape
        
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val, size=size, dtype=self.dtype)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(size) + [NUM_OF_COMPARE_BITS], dtype=np.int8)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val, size=size, dtype=self.dtype)
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=size,dtype=self.dtype)
        self.private_compare(dummy_tensor)
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=size, dtype=self.dtype)
        self.mult(dummy_tensor)

        return dummy_tensor


class PRFFetcherDReLU(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherDReLU, self).__init__(crypto_assets, network_assets)

        self.share_convert = PRFFetcherShareConvert(crypto_assets, network_assets)
        self.msb = PRFFetcherMSB(crypto_assets, network_assets)

    def forward(self, dummy_tensor):

        self.share_convert(dummy_tensor)
        self.msb(dummy_tensor)

        return dummy_tensor


class PRFFetcherReLU(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets, dummy_relu=False):
        super(PRFFetcherReLU, self).__init__(crypto_assets, network_assets)

        self.DReLU = PRFFetcherDReLU(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)
        self.dummy_relu = dummy_relu

    def forward(self, dummy_tensor):
        if self.dummy_relu:
            return dummy_tensor
        else:
            dummy_arr = dummy_tensor.astype(self.dtype).flatten()
            self.DReLU(dummy_arr)
            self.mult(dummy_arr)
            return dummy_tensor


class PRFFetcherBlockReLU(PRFFetcherModule):

    def __init__(self, crypto_assets, network_assets, block_sizes, dummy_relu=False):
        super(PRFFetcherBlockReLU, self).__init__(crypto_assets, network_assets)
        self.block_sizes = np.array(block_sizes)
        self.dummy_relu = dummy_relu

        self.DReLU = PRFFetcherDReLU(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if
                                   0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def forward(self, dummy_tensor):
        if self.dummy_relu:
            return dummy_tensor
        dummy_arr = dummy_tensor
        mean_tensors = []

        for block_size in self.active_block_sizes:
            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = dummy_arr[:, cur_channels]
            reshaped_input = SpaceToDepth(block_size)(cur_input)
            mean_tensor = np.sum(reshaped_input, axis=-1, keepdims=True)

            mean_tensors.append(mean_tensor.flatten())

        mean_tensors = np.concatenate(mean_tensors)

        self.DReLU(mean_tensors.astype(self.dtype))
        self.mult(dummy_arr[:, ~self.is_identity_channels].astype(self.dtype))

        return dummy_tensor

class PRFFetcherSecureModel(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(PRFFetcherSecureModel, self).__init__( crypto_assets, network_assets)
        self.model = model

    def forward(self, img):
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(low=MIN_VAL // 2, high=MAX_VAL // 2, size=img.shape, dtype=SIGNED_DTYPE)
        out_0 = self.model.decode_head(self.model.backbone(np.zeros(shape=img.shape, dtype=SIGNED_DTYPE)))


