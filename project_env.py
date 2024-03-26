# coding=utf-8
# ----------------------------------------------------------------
# @time: 2024/03/15 01:28
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
import logging
import os

from facechain.wktk.base_utils import PF

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PF.p(f"[os.path.abspath(./)] {os.path.abspath('./')}", layer_back=1)
PF.p(f'[        PROJECT_DIR] {PROJECT_DIR}', layer_back=1)
PF.p(f"[           __file__] {__file__}", layer_back=1)
PF.p('\n' * 20)

# MODELSCOPE_CACHE = f"{PROJECT_DIR}/data/cache_model/modelscope/hub"
MODELSCOPE_CACHE = f"./"  # 避免预训练模型里包含其他预训练模型(wrapper.py)导致下载后路径加载失败
os.environ["MODELSCOPE_CACHE"] = MODELSCOPE_CACHE


def config_logger(log_name='root', log_file=None):
    """https://stackoverflow.com/a/44296581/6494418
    # modelscope getLogger
        - modelscope.utils.logger.get_logger
        - accelerate.logging.get_logger
    """

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    # https://stackoverflow.com/a/533077/6494418
    # log_format = "[%(asctime)s] [{N}] %(pathname)s %(module)s \"%(filename)s:%(lineno)d\" %(message)s".format(N=log_name)
    log_format = "[%(asctime)s] [{N}] \"%(pathname)s:%(lineno)d\" %(message)s".format(N=log_name)
    formatter = logging.Formatter(log_format)
    logging.basicConfig(format=log_format)

    class NoDeprecatedFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return (
                    "is deprecated" not in msg
                    and "pthread_setaffinity_np failed for thread" not in msg
            )

    logger.addFilter(NoDeprecatedFilter())

    # create file handler which logs even debug messages
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    PF.p(f'[config_logging] {log_name}', layer_back=2)


def init_env():
    config_logger('root')  # worked
    config_logger('modelscope')  # not work with base??
