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
os.environ["MODELSCOPE_CACHE"] = f"{project_dir}/input/cache/modelscope/hub"
print('[os.environ["MODELSCOPE_CACHE"]] ', os.environ["MODELSCOPE_CACHE"])
print('[project_dir] ', project_dir)


def config_logging(log_name='root', log_file=None):
    """https://stackoverflow.com/a/44296581/6494418
    # modelscope getLogger
        - modelscope.utils.logger.get_logger
        - accelerate.logging.get_logger
    """

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("[%(asctime)s] [{N}] \"%(filename)s:%(lineno)d\" %(message)s".format(N=log_name))

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
    config_logging('root')
    config_logging('modelscope')
