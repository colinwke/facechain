# coding=utf-8
# ----------------------------------------------------------------
# @time: 2024/03/15 01:28
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
import logging
import os

from facechain.wktk.base_utils import PF

project_dir = os.path.dirname(os.path.abspath(__file__))

PF.p('\n' * 20)
# os.environ["MODELSCOPE_CACHE"] = f"{project_dir}/input/cache2/modelscope/hub"
# os.environ["MODELSCOPE_CACHE"] = f"{project_dir}/input/cache"
print('[project_dir] ', project_dir)


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
    log_format = "[%(asctime)s] [{N}] %(pathname)s %(module)s \"%(filename)s:%(lineno)d\" %(message)s".format(N=log_name)
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

    print('[config_logging] ', log_name)


def init_env():
    config_logger('root')  # worked
    config_logger('modelscope')  # not work with base??
