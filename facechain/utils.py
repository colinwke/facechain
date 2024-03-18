# Copyright (c) Alibaba, Inc. and its affiliates.

import multiprocessing as mp
import os
import subprocess
import time

from modelscope import snapshot_download as ms_snapshot_download

from facechain.wktk.base_utils import PF

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PF.p(f'[project_dir] {project_dir}')
PF.p(f"[os.path.abspath(./)] {os.path.abspath('./')}")
PF.p(f"[__file__] {__file__}")


# os.environ["MODELSCOPE_CACHE"] = f"{project_dir}/input/cache/modelscope/hub3"


def max_retries(max_attempts):
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"[max_retries] Retry {attempts}/{max_attempts}: {e}")
                    time.sleep(1)  # wait 1 sec
            raise Exception(f"[max_retries] Max retries ({max_attempts}) exceeded.")

        return wrapper

    return decorator


@max_retries(1)
def snapshot_download_dk(model_id, revision, cache_dir=None, user_agent=None):
    # from modelscope.hub.snapshot_download import snapshot_download
    PF.p('[snapshot_download_dk.1]', model_id, revision, cache_dir, layer_back=2)
    # /code/dkc/facechain-main/input/cache/modelscope/hub/Cherrytest/rot_bgr
    # model_dir = f'{project_dir}/input/cache/modelscope/hub/{model_id}'
    # WPAI离线训练不支持访问外网, 抛出错误`Network is unreachable`, 则直接使用本地路径
    cache_dir = None
    cache_dir = f'{project_dir}/input/cache/modelscope/hub/'  # 如果没哟指定`cache_dir`, 则会生成读取默认的`cache_dir`, 其中的一部分包含了`MODELSCOPE_CACHE`
    # cache_dir = f'./input4/cache/modelscope/hub/'  # 如果没哟指定`cache_dir`, 则会生成读取默认的`cache_dir`, 其中的一部分包含了`MODELSCOPE_CACHE`
    # cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../', 'input3/cache/modelscope/hub')

    if 'MODELSCOPE_CACHE' in os.environ:
        print('[os.environ["MODELSCOPE_CACHE"]] ', os.environ["MODELSCOPE_CACHE"])

    model_dir = ms_snapshot_download(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        user_agent=user_agent,
        local_files_only=False
    )
    # try:
    #     pass
    # except:
    #     pass
    PF.p('[snapshot_download_dk.2] model_dir', model_dir, layer_back=2)
    return model_dir


@max_retries(1)
def snapshot_download_dk1(*args, **kwargs):
    PF.p('snapshot_download_dk', str(args), str(kwargs), layer_back=2)
    return ms_snapshot_download(*args, **kwargs)


def pre_download_models():
    snapshot_download_dk('ly261666/cv_portrait_model', revision='v4.0')
    snapshot_download_dk('YorickHe/majicmixRealistic_v6', revision='v1.0.0')
    snapshot_download_dk('damo/face_chain_control_model', revision='v1.0.1')
    snapshot_download_dk('ly261666/cv_wanx_style_model', revision='v1.0.3')
    snapshot_download_dk('damo/face_chain_control_model', revision='v1.0.1')
    snapshot_download_dk('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
    snapshot_download_dk('Cherrytest/rot_bgr', revision='v1.0.0')
    snapshot_download_dk('damo/face_frombase_c4', revision='v1.0.0')


def set_spawn_method():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("spawn method already set")


def check_install(*args):
    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT)
        return True
    except OSError as e:
        return False


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    return check_install("ffmpeg", "-version")


def get_worker_data_dir() -> str:
    """Get the worker data directory."""
    return os.path.join(project_dir, "worker_data")


def join_worker_data_dir(*kwargs) -> str:
    """Join the worker data directory with the specified sub directory."""
    return os.path.join(get_worker_data_dir(), *kwargs)
