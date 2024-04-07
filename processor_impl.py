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
import os
import os.path
import threading
from importlib import reload
from time import sleep

import env_checker

reload(env_checker)
from env_checker import check_env

check_env()  # check mmcv etc. first!

from facechain.train_text_to_image_lora import main_training

import facechain.wktk.base_utils

reload(facechain.wktk.base_utils)
from facechain.wktk.base_utils import *
from run_inference import main_predict

WOS_BUKET_ID = "imgfaceid"
WOS_URL_PREFIX = f'http://prod17.wos.58dns.org/QrxnMlKjQrtW/{WOS_BUKET_ID}'
MAX_JOB_COUNT = 2
MIN_GPU_FREE = 1000
thread_manager = []
DEFAULT_DES = 'formal wear, formal clothes, identification photo, ID photo, raw photo, masterpiece, chinese, __prompt_cloth__, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, photorealistic, best quality'


def make_ret_val(req_dict, code=-999, msg='unknown', result_file=None):
    """ ret_dict.update(kwargs) """
    ret_dict = {
        'ts': req_dict.get('ts'),
        'sid': req_dict.get('sid'),
        'code': str(code),
        'msg': msg,
    }
    ret_dict.update(req_dict)
    ret_val = UT2.json2str(ret_dict)
    PF.p(f'[ret_val] {ret_val}', layer_back=1)
    with open(f'ret_msg_{UT2.get_ts(tm=False)}.log', 'a') as f:
        f.write(f'{ret_val}\n')

    # 单独写入result_file, 并返回wos上的地址, 这里就能保证上传wos的结果文件与日志文件一致
    if result_file is not None:
        with open(result_file, 'w+') as f:
            f.write(f"{ret_val}\n")
            UT2.upload_to_wos(result_file, None, WOS_BUKET_ID)

    return ret_val


def run_model_impl(model, x, kwargs={}, **kwargs2):
    """ 多线程处理框架, 达到请求即返回结果的目的 """
    global thread_manager
    thread_manager = [t for t in thread_manager if t.is_alive()]
    job_count = len(thread_manager) + 1  # 当前任务算上thread
    gpu_free = UT2.get_gpu_info().free
    req_dict = json.loads(x)

    if 'sid' not in req_dict or len(str(req_dict['sid'])) < 5:
        req_dict['sid'] = str(UT2.get_uuid())
    if 'des' not in req_dict or len(str(req_dict['des'])) < 5:
        req_dict['des'] = DEFAULT_DES

    url = req_dict.get('pic', 'pic url is empty!')
    url_available = UT2.check_url(url)
    ts = UT2.if_not_exist_put(req_dict, 'ts', UT2.get_ts())
    sid = req_dict.get('sid')
    reqid = UT2.clean_path_name(f"{req_dict['ts']}_{req_dict['sid']}")
    reqid = UT2.if_not_exist_put(req_dict, 'reqid', reqid)  # running result id
    result_file_url = UT2.if_not_exist_put(req_dict, 'result_file_url', f'{WOS_URL_PREFIX}/result_{reqid}.json')

    PF.p(f"""
    ----------------------------------------------------------------
    [CHECK_URL]   {url_available} / {url}
    [JOB_COUNT]   {job_count} / {MAX_JOB_COUNT}
    [GPU_FREE]    {gpu_free} / {MIN_GPU_FREE}
    [RESULT_FILE] {result_file_url}
    [reqid]       {reqid}
    [sid]         {sid}
    [ts]          {ts}
    ----------------------------------------------------------------
    """)

    if not url_available:
        return make_ret_val(req_dict, 1, f"链接解析出错( {url} )")
    if gpu_free <= MIN_GPU_FREE:
        return make_ret_val(req_dict, 2, f"GPU资源已被占满,稍后再试( {gpu_free}/{MIN_GPU_FREE} )")
    if job_count > MAX_JOB_COUNT:  # 这里计算的是已经存在的线程, 这个线程还没有创建
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
    tm = TimeMarker('run_model')
    des = req_dict['des']
    url = req_dict['pic']
    reqid = req_dict['reqid']

    cache_reqid_path = os.path.join("./data/cache_req", reqid)
    os.makedirs(cache_reqid_path, exist_ok=True)

    try:
        # 下载图片
        input_img_dir = os.path.join(cache_reqid_path, 'input_img')
        os.makedirs(input_img_dir, exist_ok=True)
        UT2.sh(f'curl {url} > {input_img_dir}/input_img_{reqid}.png')

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
                    UT2.upload_to_wos(filepath, filename, bucket=WOS_BUKET_ID)

        ## update to wos, 返回wos上的文件名称
        result = [f'{WOS_URL_PREFIX}/{x}' for x in result]
        PF.print_list(result, 'final_result')
        req_dict['result'] = ','.join(result)
        msg = "running success!"
        code = 101
    except Exception as e:
        msg = PF.print_stack(e=e, ret_ol=True)
        msg = f'Exception2 -- {msg}'
        code = 102

    elapse = tm.end(info='full running end')
    req_dict['elapse'] = f'{elapse:.2f}'
    result_file_name = req_dict['result_file_url'].split('/')[-1]
    result_file_path = os.path.join(cache_reqid_path, result_file_name)
    return make_ret_val(req_dict, code=code, msg=msg, result_file=result_file_path)


def local_test():
    """
    请求包含3个部分:
        1. 用户reqid(必须: 创建对应缓存文件夹, 保留24小时, 如果不改变相片则不再训练用户lora, 只进行预估)
        2. 描述(必须) 证件照, 工装照等
        3. 相片(非必须, 第一次必须)
        先快速上线demo, 后优化
    """

    for i in range(1):
        sid = UT2.get_uuid()
        sid = 'iShot_2024-03-20_18.53.51.png'
        req_dict = {
            'pic': f'http://10.186.8.94:8000/{sid}',
            'des': DEFAULT_DES,
            'sid': sid,
        }

        req_json = UT2.json2str(req_dict)
        PF.p(req_json)
        ret = run_model_impl(None, req_json)
        PF.p(ret)
        PF.p("done")

        sleep(1)

    sleep(60 * 10)  # avoid main thread close ## RuntimeError: cannot schedule new futures after interpreter shutdown


if __name__ == '__main__':
    local_test()
