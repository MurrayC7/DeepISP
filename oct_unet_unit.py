import tensorflow as tf
from .tf_cnn_basic import *
from .tf_octConv import *


def Unet_Unit_norm(data, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    conv_m1 = Conv_BN_AC(data=data, num_filter=num_mid, kernel=(1, 1), pad='valid', name=('%s_conv-m1' % name))

    return


def Unet_Unit_last(hf_data, lf_data, alpha, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    pass


def Unet_Unit_first(data, alpha, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    pass


def Unet_Unit(hf_data, lf_data, alpha, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    pass



