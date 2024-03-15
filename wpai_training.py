# coding=utf-8
# ----------------------------------------------------------------
# @time: 2024/03/12 17:11
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
import shutil
from multiprocessing import freeze_support

from facechain.train_text_to_image_lora import main_training
from run_inference import main_predict

if __name__ == '__main__':
    freeze_support()

    # shutil.rmtree('./', ignore_errors=True)
    # os.makedirs(dirname(log_file), exist_ok=True)

    print("wpai_training.py")
    main_training()
    main_predict()
