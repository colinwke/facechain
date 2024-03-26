# Copyright (c) Alibaba, Inc. and its affiliates.

import multiprocessing as mp
import os
import subprocess
import time
from typing import Optional

from modelscope import snapshot_download as ms_snapshot_download, Pipeline, pipeline
from modelscope.utils.constant import DEFAULT_MODEL_REVISION

from facechain.wktk.base_utils import PF
from project_env import PROJECT_DIR


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


# @max_retries(1)
def snapshot_download_dk(model_id, revision=DEFAULT_MODEL_REVISION, cache_dir=None, user_agent=None):
    local_file_only = not os.path.exists('/code/dkc/project/facegen_tc201')
    local_file_only = True
    model_dir = ms_snapshot_download(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        user_agent=user_agent,
        local_files_only=local_file_only
    )
    PF.p(f'[snapshot_download_dk.2] local_file_only( {local_file_only} ), model_dir( {model_dir} )', layer_back=2)
    # PF.print_stack()
    return model_dir


def pipeline_dk(
        task: str = None,
        model: str = None,
        device: str = 'gpu',
        model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
        **kwargs
) -> Pipeline:
    """overwriting for pre download model
    TODO: 加载模型变成一次加载! 非每次请求加载
    """
    model_dir = snapshot_download_dk(model, revision=model_revision)
    return pipeline(task=task, model=model_dir, model_revision=model_revision, device=device, **kwargs)


def pre_download_models():
    snapshot_download_dk('ly261666/cv_portrait_model', revision='v4.0')
    snapshot_download_dk('YorickHe/majicmixRealistic_v6', revision='v1.0.0')
    snapshot_download_dk('damo/face_chain_control_model', revision='v1.0.1')
    snapshot_download_dk('ly261666/cv_wanx_style_model', revision='v1.0.3')
    snapshot_download_dk('damo/face_chain_control_model', revision='v1.0.1')
    snapshot_download_dk('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
    snapshot_download_dk('Cherrytest/rot_bgr', revision='v1.0.0')
    snapshot_download_dk('damo/face_frombase_c4', revision='v1.0.0')

    # for predict
    snapshot_download_dk('damo/cv_manual_face-quality-assessment_fqa', model_revision='v2.0')


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
    return os.path.join(PROJECT_DIR, "worker_data")


def join_worker_data_dir(*kwargs) -> str:
    """Join the worker data directory with the specified sub directory."""
    return os.path.join(get_worker_data_dir(), *kwargs)
