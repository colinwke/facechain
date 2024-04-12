# coding=utf-8
# ----------------------------------------------------------------
# @time: 2024/04/03 14:12
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
"""
Error: nms_impl: implementation for device cuda:0 not found.
https://github.com/open-mmlab/mmdetection/issues/6765
https://blog.csdn.net/qq_43743827/article/details/126859013
重新安装mmcv-full
"""
import os
from importlib import reload

from facechain.wktk.base_utils import PF


def check_mmcv():
    try:
        os.system("yes | pip3 uninstall mmcv-full")
    except Exception as e:
        PF.print_stack(e=e)
    os.system(f"pip3 install mmcv_full")
    # os.system(f"pip3 install mmcv_full-1.7.2-cp310-cp310-linux_x86_64.whl")


def check_mmcv2():
    try:
        import torch
        from mmcv.ops import batched_nms
        device = torch.device('cuda:0')
        bboxes = torch.randn(2, 4, device=device)
        scores = torch.randn(2, device=device)
        labels = torch.zeros(2, dtype=torch.long, device=device)
        det_bboxes, keep_idxs = batched_nms(
            bboxes.to(torch.float32),
            scores.to(torch.float32),
            labels,
            {
                'type': 'nms',
                'iou_threshold': 0.6
            }
        )

        print('[check_mmcv] OK!')
    except Exception as e:
        PF.print_stack(e=e)
        print('[check_mmcv] install mmcv-full ...')
        os.system("yes | pip3 uninstall mmcv-full")
        # os.system("pip3 install mmcv-full")
        os.system(f"pip3 install mmcv_full-1.7.2-cp310-cp310-linux_x86_64.whl")
        # os.system(f"pip3 install {PROJECT_DIR}/mmcv_full-1.7.2-cp310-cp310-linux_x86_64.whl")
        import mmcv.ops
        reload(mmcv.ops)  # reload with changed # vim /usr/local/lib/python3.10/dist-packages/mmcv/ops/nms.py
        from mmcv.ops import nms
        reload(nms)
        # import importlib
        # for module in sys.modules.values():
        #     print(f'[check_mmcv] {module}')
        #     importlib.reload(module)

        # # https://stackoverflow.com/a/65156135/6494418
        # import code
        # # get source from file as a string
        # src_code = ''.join(open('/usr/local/lib/python3.10/dist-packages/mmcv/ops/nms.py').readlines())
        # # compile the source
        # compiled_code = code.compile_command(source=src_code, symbol='exec')
        # # run the code
        # eval(compiled_code)  # or exec(compiled_code)


def check_env():
    check_mmcv()


if __name__ == '__main__':
    check_mmcv()
