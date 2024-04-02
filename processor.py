# coding=utf-8
# ----------------------------------------------------------------
# @time: 2024/03/19 15:39
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
"""
只包含执行框架, 保证`seldon-core`进程的正常执行! (因为seldon-core相当于守护进程, kill了就停止了@chenzelong)
当`processor_impl_XX.py`修改了, 重新加载程序! 达到程序的无缝调试运行! (注意: 该项目不能包含argparser, 和`predictor.py`)
"""
import os

from importlib import reload


def print_stack():
    from traceback import extract_stack, format_list, format_exc
    fmt_stack_str = ''.join(format_list(extract_stack())[:-1])
    fmt_exc_str = format_exc().strip()
    fmt_stack_str = f'{fmt_stack_str}{fmt_exc_str if fmt_exc_str != "NoneType: None" else ""}'.rstrip().split('\n')
    fmt_stack_str = '\n'.join([x.replace('File "', '"').replace('", line ', ':').replace(', in ', '" -- ') for x in fmt_stack_str if x.startswith('  File "')])
    content = f"--- [print_stack] ---\n{fmt_stack_str}\n^^^^^^\n"
    print(content)
    return content


class Demo:
    """do not remove! wpai predictor.py will call! """

    def to(self, device):
        pass

    def eval(self):
        pass


def load_model():
    """load customer model, do not remove! wpai predictor.py will call! """
    return Demo()


def preprocess(x):
    """do not remove! wpai predictor.py will call! """
    return x


def postprocess(x):
    """do not remove! wpai predictor.py will call! """
    return x


IMPL_FILE = 'processor_impl'
IMPL_FILE_MFT = None
run_model_impl = None


def run_model(model, x, **kwargs):
    """run customer model for one request, do not remove! wpai predictor.py will call!
    1. 先判断impl是否存在, 如果不存在, 打印错误提示信息
    2. impl是否加载, 如果没有加载, 加载?(可直接看modification time)
    3. 每次请求都看是否文件被改变
    """
    global run_model_impl
    global IMPL_FILE_MFT

    try:
        IMPL_FILE_MFT_cur = os.path.getmtime(IMPL_FILE + '.py')

        if IMPL_FILE_MFT is None:
            import processor_impl
            IMPL_FILE_MFT = IMPL_FILE_MFT_cur
            print(f'\n\n\n\n\n[check_reload_impl_file1] update:{IMPL_FILE_MFT} to:{IMPL_FILE_MFT_cur}')
            from processor_impl import run_model_impl
        elif IMPL_FILE_MFT != IMPL_FILE_MFT_cur:
            import processor_impl
            IMPL_FILE_MFT = IMPL_FILE_MFT_cur
            print(f'\n\n\n\n\n[check_reload_impl_file2] update:{IMPL_FILE_MFT} to:{IMPL_FILE_MFT_cur}')
            reload(processor_impl)  # not update function
            from processor_impl import run_model_impl

        # noinspection PyCallingNonCallable
        ret = run_model_impl(model, x, **kwargs)
        return str(run_model_impl(model, x, **kwargs))
    except:
        return print_stack()


def t1():
    from time import sleep
    for i in range(100):
        run_model(1, 2)
        sleep(10)


if __name__ == '__main__':
    t1()
