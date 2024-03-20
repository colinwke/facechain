# coding=utf-8
# ----------------------------------------------------------------
# @time: 2024/03/19 15:39
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
"""


"""
import os.path
import re
import shutil


def load_model():
    return "None"


def run_model(model, x, **kwargs):
    pass


def clean_path(path):
    """https://stackoverflow.com/a/13593932/6494418"""
    return re.sub(r'[^\w_. -]+', '_', path).replace(".", "__")


def process(imei, des, pic_url):
    """TODO: 上传的相片的格式是什么(url, 还是byte, 还是存储的图片)?"""
    imei = clean_path(imei)
    imei_cache_path = os.path.join("./data/cache_imei", imei)  # TODO:  path 添加 datetime 前缀

    if pic_url and os.path.exists(imei_cache_path):
        shutil.rmtree(imei_cache_path)

    os.makedirs(imei_cache_path, exist_ok=True)
    os.system(f'curl "{pic_url}" > {imei_cache_path}/input_img')

    model = load_model()
    inputs_args = {
        'imei': imei,
        'des': des,
        'pic': pic_url,
    }
    run_model(model, inputs_args)

    # TODO: 返回结果


def local_test():
    """
    请求包含3个部分:
        1. 用户imei(必须: 创建对应缓存文件夹, 保留24小时, 如果不改变相片则不再训练用户lora, 只进行预估)
        2. 描述(必须) 证件照, 工装照等
        3. 相片(非必须, 第一次必须)
        先快速上线demo, 后优化
    """
    model = load_model()
    for i in range(1):
        # run_model(model,"http://10.186.8.103:8994/truck.png?points_mode=1,1&points_coord=250,184|562,290")
        run_model(model, "http://10.186.8.94:8189/proxy/8096/iShot_2024-03-20_15.36.26.png?points_mode=1,1,1&points_coord=533,556|798,675|1097,861&sid=1234567")
        print("done")


if __name__ == '__main__':
    local_test()
