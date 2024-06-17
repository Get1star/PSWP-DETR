import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create

cfg_path = './configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
cfg = load_config(cfg_path)
model = create(cfg.architecture)

blob = {
    'image': paddle.randn([1, 3, 640, 640]),
    'im_shape': paddle.to_tensor([[640, 640]]),
    'scale_factor': paddle.to_tensor([[1., 1.]])
}
paddle.flops(model, None, blob, custom_ops=None, print_detail=False)