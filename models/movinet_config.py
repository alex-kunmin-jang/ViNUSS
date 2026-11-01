"""
Inspired by
https://github.com/PeizeSun/SparseR-CNN/blob/dff4c43a9526a6d0d2480abc833e78a7c29ddb1a/detectron2/config/defaults.py
https://github.com/Atze00/MoViNet-pytorch/blob/main/movinets/models.py
"""
from fvcore.common.config import CfgNode as CN

def fill_SE_config(conf, input_channels, 
                    out_channels, 
                    expanded_channels,
                    kernel_size,
                    stride,
                    padding,
                    padding_avg,
):
    conf.expanded_channels =expanded_channels
    conf.padding_avg= padding_avg
    fill_conv(conf,input_channels,
                out_channels, 
                kernel_size,
                stride,
                padding,
)

def fill_conv(conf, input_channels,
                out_channels, 
                kernel_size,
                stride,
                padding,):
    conf.input_channels = input_channels
    conf.out_channels = out_channels
    conf.kernel_size = kernel_size
    conf.stride = stride
    conf.padding = padding
   

def get_movinet_cfg():
    return _C.clone()

_C = CN()

_C.MODEL = CN()


###############################################################################################################################################################################################################################################################################################
#### PTE_MoViNetA2 ############################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################################


_C.MODEL.PTE_MoViNetA2 = CN()
_C.MODEL.PTE_MoViNetA2.name = "PTE_MoViNetA2"
_C.MODEL.PTE_MoViNetA2.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA2_statedict_v3?raw=true"
_C.MODEL.PTE_MoViNetA2.stream_weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA2_stream_statedict_v3?raw=true"

_C.MODEL.PTE_MoViNetA2.conv000 = CN()
_C.MODEL.PTE_MoViNetA2.conv001 = CN()
_C.MODEL.PTE_MoViNetA2.conv002 = CN()
_C.MODEL.PTE_MoViNetA2.conv010 = CN()
_C.MODEL.PTE_MoViNetA2.conv011 = CN()
_C.MODEL.PTE_MoViNetA2.conv012 = CN()
_C.MODEL.PTE_MoViNetA2.convpool = CN()


# PTE branch 1 (temporal kernel 3): 1 -> 3 -> 5 -> 5 channels, 150 x 224 x 224
fill_conv(_C.MODEL.PTE_MoViNetA2.conv000, 1,3,(3,1,1),(1,1,1),(1,0,0))
fill_conv(_C.MODEL.PTE_MoViNetA2.conv001, 3,5,(3,1,1),(1,1,1),(1,0,0))
fill_conv(_C.MODEL.PTE_MoViNetA2.conv002, 5,5,(3,1,1),(1,1,1),(1,0,0))

# PTE branch 2 (temporal kernel 5): 1 -> 3 -> 5 -> 5 channels, 150 x 224 x 224
fill_conv(_C.MODEL.PTE_MoViNetA2.conv010, 1,3,(5,1,1),(1,1,1),(2,0,0))
fill_conv(_C.MODEL.PTE_MoViNetA2.conv011, 3,5,(5,1,1),(1,1,1),(2,0,0))
fill_conv(_C.MODEL.PTE_MoViNetA2.conv012, 5,5,(5,1,1),(1,1,1),(2,0,0))

# Concatenate input + both branches: 1 + 5 + 5 = 11 x 150 x 224 x 224
# Strided temporal projection: 11 x 150 -> 11 x 50 x 224 x 224
fill_conv(_C.MODEL.PTE_MoViNetA2.convpool, 11,11,(3,1,1),(3,1,1),(0,0,0))

_C.MODEL.PTE_MoViNetA2.conv1 = CN()

fill_conv(_C.MODEL.PTE_MoViNetA2.conv1, 11,24,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.PTE_MoViNetA2.blocks = [ [CN() for _ in range(3)],
                              [CN() for _ in range(5)],
                              [CN() for _ in range(5)],
                              [CN() for _ in range(6)],
                              [CN() for _ in range(7)]]

#Block2
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[0][0], 24, 24, 40, (1,5,5), (1,2,2), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[0][1], 24, 24, 40, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[0][2], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 3
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[1][0], 24, 40, 96, (3,3,3), (1,2,2), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[1][1], 40, 40, 120, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[1][2], 40, 40, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[1][3], 40, 40, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[1][4], 40, 40, 120, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 4
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[2][0], 40, 72, 240, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[2][1], 72, 72, 160, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[2][2], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[2][3], 72, 72, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[2][4], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 5
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[3][0], 72, 72, 240, (5,3,3), (1,1,1), (2,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[3][1], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[3][2], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[3][3], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[3][4], 72, 72, 144, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[3][5], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 6
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[4][0], 72 , 144, 480, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[4][1], 144, 144, 384, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[4][2], 144, 144, 384, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[4][3], 144, 144, 480, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[4][4], 144, 144, 480, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[4][5], 144, 144, 480, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.PTE_MoViNetA2.blocks[4][6], 144, 144, 576, (1,3,3), (1,1,1), (0,1,1), (0,1,1))

_C.MODEL.PTE_MoViNetA2.conv7= CN()
fill_conv(_C.MODEL.PTE_MoViNetA2.conv7, 144,640,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.PTE_MoViNetA2.dense9= CN()
_C.MODEL.PTE_MoViNetA2.dense9.hidden_dim = 2048
