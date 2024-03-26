# coding=utf-8
# ----------------------------------------------------------------
# @time: 2024/03/19 15:39
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
"""

    # freeze_support()
    # shutil.rmtree('./', ignore_errors=True)
    # os.makedirs(dirname(log_file), exist_ok=True)

"""
import json
import os.path
import re
from time import gmtime

from facechain.train_text_to_image_lora import main_training
from facechain.wktk.base_utils import *
from run_inference import main_predict


def sh(cmd):
    PF.p(f'$$ {cmd}', layer_back=1)
    os.system(cmd)


def json2str(obj):
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)


def clean_path(path):
    """https://stackoverflow.com/a/13593932/6494418"""
    return re.sub(r'[^\w_. -]+', '_', path).replace(".", "__")


def preprocess(x):
    return x


def postprocess(x):
    return x


class Demo:
    def to(self, device):
        pass

    def eval(self):
        pass


def load_model():
    return Demo()


def run_model(model, x, **kwargs):
    ts = Timestamp('run_model')
    PF.p(x)
    req = json.loads(x)
    imei = req['imei']

    tx = strftime("%y%m%d_%H%M%S", gmtime())
    imei = f"{tx}__{imei}"
    des = req['des']
    pic_url = req['pic']

    imei = clean_path(imei)
    imei_cache_path = os.path.join("./data/cache_imei", imei)  # TODO:  path 添加 datetime 前缀

    # if pic_url and os.path.exists(imei_cache_path):
    # shutil.rmtree(imei_cache_path)

    # 下载图片
    os.makedirs(os.path.join(imei_cache_path, 'input_img'), exist_ok=True)
    sh(f'curl {pic_url} > {imei_cache_path}/input_img/000.png')

    model = load_model()
    inputs_args = {
        'imei': imei,
        'des': des,
        'pic': pic_url,
    }

    print("wpai_training.py")
    main_training(imei)
    main_predict(imei, user_prompt_cloth=des)

    bucket = 'imgfaceid'

    output_dir = f'{imei_cache_path}/output_generated/'
    PF.p(f'[output_dir] {output_dir}')
    result = []
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.endswith('.png'):
                result.append(filename)
                sh(f'sh wos_client.sh -t token.wos.58dns.org -h wosin17.58corp.com -a QrxnMlKjQrtW -s gK9T1G9BBox7Tk9dW7kRtyuvXLNVuyly -b {bucket} -f {filename} -l {filepath} upload 2>&1 >/dev/null')

    ## update to wos
    ts.end(info='full running end')
    ## 返回wos上的文件名称
    result = [f'http://prod17.wos.58dns.org/QrxnMlKjQrtW/imgfaceid/{x}' for x in result]
    PF.print_list(result, 'final_result')
    ret = json2str(result)
    return ret


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
        imei = 'iShot_2024-03-20_18.52.38.png'
        imei = 'iShot_2024-03-20_18.53.51.png'
        req = {
            'pic': f'http://10.186.8.94:8000/{imei}',
            'imei': imei,
            'des': 'formal wear, formal clothes, identification photo, ID photo, raw photo, masterpiece, chinese, __prompt_cloth__, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, photorealistic, best quality'
        }

        req_json = json2str(req)
        PF.p(req_json)
        ret = run_model(model, req_json)
        print(ret)
        print("done")


if __name__ == '__main__':
    local_test()
