# coding=utf-8
# ----------------------------------------------------------------
# @time: 2024/03/19 15:39
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
"""
https://docs.58corp.com/#/space/1772520715837288448 @zhukun02
1.平台调用wpai提交生成证件照任务
(当前模型执行任务耗时约3分钟, 所以需要异步调用, 由于wpai不支持异步, 故模型内部自己实现异步)

2.wpai同步返回任务提交结果
 举例:
 code:1 链接解析出错
 code:2 GPU资源已被占满,稍后再试
 code:3 内部执行出错
 code:4 任务提交成功

3.平台根据状态码, 返回结果给公众号
---

平台请求模型会传一个请求id, 此id需要唯一
结果文件格式约定, 文件名格式: 请求id_result.json
文件内容:
{
    "code":
    "msg":
    "fileUrlList": []
}

"""
import json
import os.path
import re
import threading
import urllib
import uuid
from time import sleep
from urllib.parse import urlparse, parse_qs

import requests
from nvitop import Device

from facechain.train_text_to_image_lora import main_training
from facechain.wktk.base_utils import *
from run_inference import main_predict


class UT:
    UNIT_MB_OF_BYTE = 1024 * 1024

    @staticmethod
    def check_url(url):
        response = requests.get(url)
        if response.status_code == 200:
            return True
        return False

    @staticmethod
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @staticmethod
    def parse_url(url_raw):
        """
        http://10.186.8.103:8994/pz.mp4?  points_mode=1,1,1&points_coord=300,300|306,249|306,359&sid=12345678910
        """
        ret_dict = {}

        url_fix = urllib.parse.unquote(url_raw)
        if not UT.is_valid_url(url_fix):
            return {"": ""}

        url_pic = url_raw.split('?')[0]
        filename = url_pic.split('/')[-1]
        url_parsed = urlparse(url_raw)
        url_args = parse_qs(str(url_parsed.query))

        def __get_url_args(key, default_val=None):
            """ uid = __get_url_args('uid', '') """
            return url_args.get(key, [default_val])[0]

        uid = __get_url_args('uid', '')

        return ret_dict

    @staticmethod
    def get_gpu_info():
        """ gpu info by nvitop(more: gpustat, pynvml).
         - https://nvitop.readthedocs.io/en/latest/#quick-start # nvitop
         - https://stackoverflow.com/a/59571639/6494418 # nvidia-smi --query-gpu=memory.free --format=csv
        """
        device0 = Device.all()[0]
        gpu_info = AttrDict({
            'utl': device0.gpu_utilization(),
            'total': device0.memory_total() / UT.UNIT_MB_OF_BYTE,
            'used': device0.memory_used() / UT.UNIT_MB_OF_BYTE,
            'free': device0.memory_free() / UT.UNIT_MB_OF_BYTE
        })

        return gpu_info

    @staticmethod
    def sh(cmd):
        PF.p(f'$$ {cmd} 2>&1 >/dev/null', layer_back=1)
        os.system(cmd)

    @staticmethod
    def json2str(obj):
        return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)

    @staticmethod
    def clean_path_name(path):
        """clean all punctuation https://stackoverflow.com/a/13593932/6494418"""
        return re.sub(r'[^\w_. -]+', '_', path).replace(".", "__")

    @staticmethod
    def get_ts(tm=True):
        """gmtime not correct with localtime. timestamp, dts:`%y%m%d_%H%M%S`; unixtimestamp, tms, ms: 12134"""
        return strftime("%y%m%d_%H%M%S" if tm else "%y%m%d", localtime())

    @staticmethod
    def get_tsd(tm=True):
        """https://stackoverflow.com/a/71079084/6494418"""
        return datetime.now().strftime("%y%m%d_%H%M%S" if tm else "%y%m%d")

    @staticmethod
    def get_uuid():
        return str(uuid.uuid4().hex)

    @staticmethod
    def N(v=None, d=None):
        return v if v is not None else d

    @staticmethod
    def if_not_exist_put(d, key, val):
        if key not in d:
            d[key] = val
            return val
        else:
            return d[key]

    @staticmethod
    def upload_to_wos(local_filepath, upload_filename, bucket):
        UT.sh(
            f'sh wos_client.sh -t token.wos.58dns.org -h wosin17.58corp.com -a QrxnMlKjQrtW -s gK9T1G9BBox7Tk9dW7kRtyuvXLNVuyly -b {bucket} -l {local_filepath} -f {upload_filename} upload'
        )


WOS_BUKET_ID = "imgfaceid"
WOS_URL_PREFIX = f'http://prod17.wos.58dns.org/QrxnMlKjQrtW/{WOS_BUKET_ID}'
MAX_JOB_COUNT = 1
MIN_GPU_FREE = 1000
thread_manager = []


def make_ret_val(req_dict, code=-999, msg='unknown'):
    """ ret_dict.update(kwargs) """
    ret_dict = {
        'ts': req_dict.get('ts'),
        'sid': req_dict.get('sid'),
        'code': str(code),
        'msg': msg,
    }
    ret_dict.update(req_dict)
    ret_val = UT.json2str(ret_dict)
    PF.p(f'[ret_val] {ret_val}', layer_back=1)
    with open(f'ret_msg_{UT.get_ts(tm=False)}.log', 'a') as f:
        f.write(f'{ret_val}\n')

    return ret_val


def run_model_impl(model, x, kwargs={}):
    """ 多线程处理框架, 达到请求即返回结果的目的 """
    global thread_manager
    thread_manager = [t for t in thread_manager if t.is_alive()]
    job_count = len(thread_manager)
    gpu_free = UT.get_gpu_info().free
    req_dict = json.loads(x)
    url = req_dict.get('pic', 'pic url is empty!')
    url_available = UT.check_url(url)
    sid = req_dict.get('sid')
    ts = UT.if_not_exist_put(req_dict, 'ts', UT.get_ts())
    reqid = f"{req_dict['ts']}_{req_dict['sid']}"  # running result id
    result_file_url = UT.if_not_exist_put(req_dict, 'result_file_url', f'{WOS_URL_PREFIX}/result_{reqid}.json')

    PF.p(f"""
    ----------------------------------------------------------------
    [CHECK_URL]   {url_available}/{url}
    [JOB_COUNT]   {job_count}/{MAX_JOB_COUNT}
    [GPU_FREE]    {gpu_free}/{MIN_GPU_FREE}
    [RESULT_FILE] {result_file_url}
    [REQ_ID]      {reqid}
    [SID]         {sid}
    [TS]          {ts}
    ----------------------------------------------------------------
    """)

    if not url_available:
        return make_ret_val(req_dict, 1, f"链接解析出错( {url} )")
    if gpu_free <= MIN_GPU_FREE:
        return make_ret_val(req_dict, 2, f"GPU资源已被占满,稍后再试( {gpu_free}/{MIN_GPU_FREE} )")
    if job_count >= MAX_JOB_COUNT:  # 这里计算的是已经存在的线程, 这个线程还没有创建
        return make_ret_val(req_dict, 2, f"任务队列已被占满,稍后再试( {job_count}/{MAX_JOB_COUNT} )")

    # 新建线程处理
    try:
        thread = threading.Thread(name="run_model_inner", target=run_model_inner, args=(req_dict,))
        thread_manager.append(thread)
        thread.start()
        # thread.join()  # join 会等待子进程运行完成后才执行后面内容 ## Error "cannot schedule new futures after interpreter shutdown" https://stackoverflow.com/a/67621024/6494418
        return make_ret_val(req_dict, 4, f"任务提交成功")
    except Exception as e:
        msg = PF.print_stack(e=e, ret_ol=True)
        return make_ret_val(req_dict, 3, f"内部执行错误 -- {msg}")


def run_model_inner(req_dict):
    try:
        tm = TimeMarker('run_model')
        des = req_dict['des']
        url = req_dict['pic']
        reqid = req_dict['reqid']

        cache_reqid_path = os.path.join("./data/cache_req", reqid)

        # 下载图片
        os.makedirs(os.path.join(cache_reqid_path, 'input_img'), exist_ok=True)
        UT.sh(f'curl {url} > {cache_reqid_path}/input_img/input_img_{reqid}.png')

        PF.p("wpai_training.py")
        main_training(reqid)
        main_predict(reqid, user_prompt_cloth=des)

        output_dir = f'{cache_reqid_path}/output_generated/'
        PF.p(f'[output_dir] {output_dir}')
        result = []
        for root, dirs, files in os.walk(output_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                if filename.endswith('.png'):
                    result.append(filename)
                    UT.upload_to_wos(filepath, filename, bucket=WOS_BUKET_ID)

        ## update to wos
        tm.end(info='full running end')
        ## 返回wos上的文件名称
        result = [f'{WOS_URL_PREFIX}/{x}' for x in result]
        PF.print_list(result, 'final_result')
        req_dict['result'] = ','.join(result)
        req_dict_json_str = UT.json2str(req_dict)
        result_file_name = req_dict['result_file_url'].split('/')[-1]
        result_file_path = os.path.join(cache_reqid_path, result_file_name)
        with open(result_file_path, 'w+') as f:
            f.write(f"{req_dict_json_str}\n")
        UT.upload_to_wos(result_file_path, result_file_name, WOS_BUKET_ID)
        return make_ret_val(req_dict, 101, "running success!")
    except Exception as e:
        msg = PF.print_stack(e=e, ret_ol=True)
        return make_ret_val(req_dict, 102, f'Exception2 -- {msg}')


def local_test():
    """
    请求包含3个部分:
        1. 用户reqid(必须: 创建对应缓存文件夹, 保留24小时, 如果不改变相片则不再训练用户lora, 只进行预估)
        2. 描述(必须) 证件照, 工装照等
        3. 相片(非必须, 第一次必须)
        先快速上线demo, 后优化
    """

    model = load_model()
    for i in range(2):
        sid = UT.get_uuid()
        # run_model(model,"http://10.186.8.103:8994/truck.png?points_mode=1,1&points_coord=250,184|562,290")
        reqid = 'iShot_2024-03-20_18.52.38.png'
        reqid = 'iShot_2024-03-20_18.53.51.png'
        req_dict = {
            'pic': f'http://10.186.8.94:8000/{reqid}',
            'reqid': reqid,
            'des': 'formal wear, formal clothes, identification photo, ID photo, raw photo, masterpiece, chinese, __prompt_cloth__, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, photorealistic, best quality',
            'sid': sid,
        }

        req_json = UT.json2str(req_dict)
        PF.p(req_json)
        ret = run_model(model, req_json)
        PF.p(ret)
        PF.p("done")

        sleep(1)

    sleep(60 * 10)  # avoid main thread close ## RuntimeError: cannot schedule new futures after interpreter shutdown


if __name__ == '__main__':
    local_test()
