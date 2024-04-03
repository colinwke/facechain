# coding=utf-8
# ----------------------------------------------------------------
# @time: 2022/04/25 17:11
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
import warnings

warnings.simplefilter('ignore')

from sys import version_info

if version_info >= (3, 10):
    from collections.abc import Iterable
else:
    from collections import Iterable

from collections import OrderedDict
from json import dumps, loads
from logging import Formatter, StreamHandler, FileHandler, INFO, getLogger, Filter
from os import makedirs, system
from os import walk as os_walk
from os.path import dirname, basename, join as path_join, exists, abspath, realpath, splitext as path_splitext, splitext

import functools
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from datetime import datetime, timedelta
from io import StringIO
from math import floor
from re import sub, search, findall, split
from requests import post
from shutil import copy2 as shutil_copy2, move as shutil_move
from shutil import rmtree
from subprocess import check_output as sp_check_output, STDOUT as SP_STDOUT, check_output, Popen, PIPE
from sys import _getframe, stdout, getsizeof, path as sys_path, argv
from time import strftime, localtime, mktime, strptime, time
from traceback import format_stack
from yaml import safe_load as load, safe_dump as dump, YAMLError

RUN_TIME = strftime("%y%m%d_%H%M%S", localtime())
SEP_LINE_64 = "-" * 64


def _is_local_running():
    local_paths = [
        "/Users/wangke/PycharmProjects/deep_ctr",
        "/home/hdp_lbg_ectech/wangke/project/deep_ctr"
    ]
    for local_path in local_paths:
        if exists(local_path):
            return True
    return False


def float_x(x):
    """`float('123_123')` under_socre will ignore: https://stackoverflow.com/a/20929881/6494418"""
    xs = str(x)
    if '_' in xs:
        raise ValueError('`_` in float value!')
    return float(x)


IS_LOCAL_RUNNING = _is_local_running()
C_FILE_ROOT_DIR = dirname(__file__)
C_FILE_BASENAME = basename(__file__)


class Logger:
    """https://stackoverflow.com/a/32001771/6494418
       https://stackoverflow.com/a/59790484/6494418"""
    _DFT_ = 'DF'
    _LOG_DICT_ = dict()

    @staticmethod
    def not_wktk_file(filename):
        """filename is relative path not absolute path"""
        if filename.endswith(C_FILE_BASENAME):  # 执行文件为本身(最高优先级)
            return True
        if 'site-packages/wktk4py' in filename:
            return False  # 包依赖wktk4py
        if filename.startswith(C_FILE_ROOT_DIR):
            return False  # 与当前脚本同文件夹
        return True

    @staticmethod
    def _get_caller(offset=4, layer_back=0):
        """Returns a code and frame object for the lowest non-logging stack frame.
        refs: tensorflow.python.platform.tf_logging._logger_find_caller
            you could answer: https://stackoverflow.com/q/24438976/6494418
                              https://stackoverflow.com/q/19615876/6494418
        """
        f = _getframe(offset)  # 这个是文件层级?, 不是调用的方法层级
        f = f.f_back  # 这个至少是打印的logging, 所以需要back一级(call inner PF.p())
        ret = (None, None)
        # print(f'--- {layer_back}')
        while f:
            code = f.f_code
            code_filename = str(code.co_filename)
            b_not_wktk_file = Logger.not_wktk_file(code_filename)
            # print(f'[layer_back_c] {layer_back:>5} {code.co_name:>30} {b_not_wktk_file!s:>5} "{code.co_filename}:{f.f_lineno}"')
            if b_not_wktk_file:  # 打印不是wktk下的stack信息
                # 在<module>(非函数里)的import调用最底层
                if '<frozen importlib' in code_filename:
                    break
                # 这里是调用栈的逆序, 如果调用上一层, 则layer_bak -1
                layer_back -= 1
                if layer_back == -1:
                    return code, f  # 返回不是base_utils下面路径文件的信息
            ret = (code, f)
            f = f.f_back
        return ret

    @staticmethod
    def _logger_find_caller(stack_info=False):
        Logger._logger_find_caller(stack_info, stacklevel=1)

    # noinspection PyRedeclaration
    @staticmethod
    def _logger_find_caller(stack_info=False, stacklevel=1):
        """refs: tensorflow.python.platform.tf_logging._logger_find_caller
                 logging.Logger._log
                 logging.Logger.findCaller
        """
        # 这里因为只有stack_info参数, 不能传入caller layer的参数(refs: logging.Logger.findCaller)
        # 所以将stack_info修改为dict
        layer_back = 0
        if isinstance(stack_info, dict):
            layer_back = stack_info.get('layer_back', 0)
            stack_info = stack_info.get('stack_info', False)
        code, frame = Logger._get_caller(offset=4, layer_back=layer_back)  # why default offset=4?
        stack_info_content = None
        if stack_info:
            stack_info_content = '\n'.join(format_stack())
        if code:
            file_name = str(code.co_filename)
            if 'ipython' in file_name:
                file_name = file_name.split('-')[2]
            return file_name, frame.f_lineno, code.co_name, stack_info_content
        else:
            return '(unknown filex)', 0, '(unknown functionx)', stack_info_content

    @staticmethod
    def get_formater(name):
        return Formatter(
            fmt="[%(asctime)s] [{N}] \"%(filename)s:%(lineno)d\" %(message)s".format(N=name),  # [%(levelname)s]
            datefmt="%y-%m-%d %H:%M:%S")

    @staticmethod
    def set_stream_handler(logger):
        handler = StreamHandler(stdout)
        handler.setFormatter(Logger.get_formater(logger.name))
        logger.addHandler(handler)
        return logger

    @staticmethod
    def set_string_io_handler(logger, string_io=None):
        if string_io:
            handler = StreamHandler(string_io)
            handler.setFormatter(Logger.get_formater(logger.name))
            logger.addHandler(handler)
        return logger

    @staticmethod
    def set_file_handler(logger, log_file=None):
        if log_file:
            dir_name = dirname(log_file)
            if dir_name != '':  # 只有文件名称(没有路径信息)时, dirname='', raise error
                makedirs(dirname(log_file), exist_ok=True)
            handler = FileHandler(log_file)
            handler.setFormatter(Logger.get_formater(logger.name))
            logger.addHandler(handler)
        return logger

    @staticmethod
    def init_tf_logger():
        class NoDeprecatedFilter(Filter):
            def filter(self, record):
                return (
                        "is deprecated" not in record.getMessage()
                        and "input must be a single string Tensor" not in record.getMessage()
                )

        logger = getLogger('tensorflow')
        if IS_LOCAL_RUNNING:
            logger.propagate = False
        logger.addFilter(NoDeprecatedFilter())

    @staticmethod
    def get_logger(name=_DFT_, log_file=None, string_io=None):
        """refs:
            https://stackoverflow.com/a/46098711
            https://stackoverflow.com/a/11233293
        """
        logger = getLogger(name)
        if name not in Logger._LOG_DICT_:
            logger.findCaller = Logger._logger_find_caller
            logger.handlers.clear()
            logger.propagate = False
            logger.setLevel(INFO)

            Logger.set_stream_handler(logger)
            Logger.set_string_io_handler(logger, string_io)
            Logger.set_file_handler(logger, log_file)
            Logger._LOG_DICT_[name] = {'log_file': log_file, "string_io": string_io}
        else:
            if string_io:
                old_string_io = Logger._LOG_DICT_[name].get('string_io')
                if not old_string_io:  # 有新的且之前未设置, 如果已经存在不设置
                    Logger.set_string_io_handler(logger, string_io)
                    Logger._LOG_DICT_[name]['string_io'] = string_io
            if log_file:
                old_log_file = Logger._LOG_DICT_[name].get('log_file')
                if old_log_file:  # 存在log_file, log_file需要保证一致性
                    assert old_log_file != log_file, "old_log_file(%s) != log_file(%s)" % (old_log_file, log_file)
                else:
                    # 不存在则添加
                    Logger.set_file_handler(logger, log_file)
                    Logger._LOG_DICT_[name]['log_file'] = log_file
        return logger

    @staticmethod
    def get_print_fn(name=_DFT_, log_file=None, string_io=None):
        # PF.p(self, *args, sep=' ', end='\n', file=None): # known special case of print
        # info(self, msg, *args, **kwargs):
        Logger.init_tf_logger()
        return Logger.get_logger(name, log_file, string_io).info


class PF:
    PRT = True  # 打印与上一次打印的间隔
    _name = 'PF'  # default name
    _p_time = time()
    _print_fn = Logger.get_print_fn(_name)  # default _print_fn

    @staticmethod
    def init(name=_name, log_file=None, string_io=False, prt=True):
        PF.p('PF.init(name=%s, string_io=%s, log_file=%s, prt=%s)' % (name, string_io is not None, log_file, prt))
        conf = Logger._LOG_DICT_.get(name)
        old_string_io = conf.get('string_io', None) if conf else None
        string_io = StringIO() if string_io is True and not old_string_io else old_string_io

        PF._name = name
        PF._print_fn = Logger.get_print_fn(name, log_file, string_io)
        PF.PRT = prt

    @staticmethod
    def get_string(name=_name):
        conf = Logger._LOG_DICT_.get(name)
        # PF.print_dict(Logger._LOG_DICT_, 'Logger._LOG_DICT_')
        string_io = conf.get('string_io') if conf else None
        return string_io.getvalue() if string_io else None

    @staticmethod
    def p(*args, sep=' ', layer_back=0, prt=None, nl=0):
        # PF.p(self, *args, sep=' ', end='\n', file=None): # known special case of print
        # PF.info(s, stack_info=dict(layer_back=1))
        stack_info = False if layer_back == 0 else dict(layer_back=layer_back)
        len_args = len(args)
        if len_args == 0:
            msg = ''
        elif len_args == 1:
            msg = str(args[0])
        else:
            msg = sep.join([str(x) for x in args])

        if nl and nl > 0:
            msg = ('\n' * nl) + msg

        msg_prt = msg
        if prt or PF.PRT:  # 打印与上一次打印的间隔
            c_time = time()
            p_runtime = c_time - PF._p_time
            PF._p_time = c_time
            msg_prt = ('[%4.2fm] ' % (p_runtime / 60)) + msg

        PF._print_fn(msg_prt, stack_info=stack_info)
        return msg

    @staticmethod
    def phf(s, le=80, fi="-"):
        msg = SU.hf(s, le, fi)
        return PF.p(msg)

    @staticmethod
    def print_block(title, content, le=80, fi="-"):
        msg = SU.hf(str(title), le, fi) + "\n" + str(content) + "\n"
        return PF.p(msg)

    @staticmethod
    def print_list(list_, title=None, fill="\n "):
        if not Checker.like_list(list_):
            temp = list_
            list_ = title
            title = temp

        title = (title if title else "print list") + (" // len=%d" % len(list_))
        _FXF = '%{N}d: %s'.format(N=len(str(len(list_))))
        msg = fill.join([SU.hf(title)] + [_FXF % x for x in enumerate(list_)]) + '\n'
        return PF.p(msg)

    @staticmethod
    def print_dict(dict_, title=None, ignore_keys=None):
        title = (title if title else "print dict") + (" // len=%d" % len(dict_))
        ignore_info = ""
        if ignore_keys is not None:
            dict_ = dict_.copy()
            ignore_info = "- ignore: " + str(ignore_keys) + "\n"
            for x in Checker.tuple(ignore_keys):
                if x in dict_:
                    dict_.pop(x)
        dict_dumps = dumps(dict_, ensure_ascii=False, indent=2, default=lambda o: str(o))
        return PF.print_block(title, ignore_info + dict_dumps)

    @staticmethod
    def print_argv():
        return PF.print_list(argv, title="PF.print_argv")

    @staticmethod
    def print_stack_format_line(stack_str):
        return '\n'.join([x.replace('File "', '"').replace('", line ', ':').replace(', in ', '" -- ')
                          if x.startswith('  File "') else x for x in stack_str.rstrip().split('\n')])

    @staticmethod
    def print_stack2():
        from traceback import extract_stack, format_list, format_exc
        fmt_stack_str = ''.join(format_list(extract_stack())[:-1])
        fmt_exc_str = format_exc().strip()
        fmt_stack_str = f'{fmt_stack_str}{fmt_exc_str if fmt_exc_str != "NoneType: None" else ""}'
        fmt_stack_str = PF.print_stack_format_line(fmt_stack_str)
        content = f"--- [print_stack] ---\n{fmt_stack_str}\n^^^^^^\n"
        print(content)
        return content

    @staticmethod
    def print_stack(title='', content='', end_layer=-1, detail=True, e=None, e_var=False, ret_ol=True):
        """https://stackoverflow.com/a/16589622/6494418"""
        from sys import exc_info
        from traceback import extract_stack, format_exc, format_list, TracebackException
        exc0 = exc_info()[0]
        stack = extract_stack()[:end_layer]  # last one would be full_stack()
        if exc0 is not None:  # i.e. an exception is present
            del stack[-1]  # remove call of full_stack, the printed exception
            # will contain the caught exception caller instead
        info_keeper = []
        title_formatted = ' // ' + title if title else ''
        if exc0 is not None:  # 是否有异常
            if detail:
                info_keeper.append(SU.hf('exception%s' % title_formatted, pre=''))
                info_keeper.append(format_exc().strip())
                info_keeper.append("^^^^^^")
            else:
                info_keeper.append('^^^^^^ ' + format_exc().strip().split('\n')[-1])
        if e and e_var:
            """https://stackoverflow.com/a/67417480/6494418"""
            # except Exception as e
            tb = TracebackException.from_exception(e, capture_locals=True)
            info_keeper.append(SU.hf('e capture_locals%s' % title_formatted, pre=''))
            info_keeper.append("".join(tb.format()).rstrip())
        if detail:
            """refs: traceback.StackSummary.format"""
            info_keeper.append(SU.hf('more_stack%s' % title_formatted, pre=''))
            info_keeper.append(PF.print_stack_format_line(''.join(format_list(stack))))
        if content:
            info_keeper.append(SU.hf('PF.exit', pre=''))
            info_keeper.append(content)
            info_keeper.append('')
        info_keeper = PF.print_stack_format_line('\n'.join(info_keeper))
        if detail:
            info_keeper = 'print_stack\n' + info_keeper
        return PF.p(info_keeper).replace('\n', ' NNNN ') if ret_ol else PF.p(info_keeper)

    @staticmethod
    def exit(*args):
        # raise ValueError(*args)  # raise Error will stop running if not try catch
        PF.print_stack(title='PF.exit', content=' '.join([str(x) for x in args]), end_layer=-2)
        exit(1018)

    @staticmethod
    def print_obj(obj):
        """https://stackoverflow.com/q/192109/6494418"""
        ret_builder = []
        ret_builder.append(PF.phf("[obj type] %s" % str(type(obj))))
        if hasattr(obj, '__dict__'):
            ret_builder.append(PF.print_dict(vars(obj), title="[vars(obj)] ++"))  # pprint(vars(obj))
        ret_builder.append(PF.print_list(dir(obj), title="[dir(obj)] ++"))  # pprint(dir(obj))
        return '\n'.join([x for x in ret_builder if x])


class LoggerKeeper:
    LOG_KEEPER = []
    _P_TIME = localtime()

    @staticmethod
    def log(info):
        c_time = localtime()
        str_c_time = DateTime.datetime(c_time, f='p')
        elapse = (mktime(c_time) - mktime(LoggerKeeper._P_TIME)) / 60
        info = "[%s %.2fm] %s" % (str_c_time, elapse, info)
        LoggerKeeper._P_TIME = c_time
        LoggerKeeper.LOG_KEEPER.append(info)
        PF.p(info)

    @staticmethod
    def get_log_keeper():
        return LoggerKeeper.LOG_KEEPER


class Checker:
    @staticmethod
    def like_list(x):
        return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

    @staticmethod
    def flatten(x):
        return [a for i in x for a in Checker.flatten(i)] if Checker.like_list(x) else [x]

    @staticmethod
    def int(v):
        """int(1.5)=1, int("1.0", base=10)=ERROR, int("1.0")=ERROR, int("1.0")=1"""
        if pd.isna(v):
            return v

        vf = float_x(v)
        if vf.is_integer():
            return int(vf)
        else:
            raise ValueError("invalid convert to int with '%s'" % v)

    @staticmethod
    def f6(x):
        return round(float_x(x), 6)

    @staticmethod
    def fn(v, n=-1):
        """1. %f会默认保留6位数字, 即使先使用了%g也只能保留6为数字; 2. 保留小数位30位(%.30f)及以上会出精度问题(20位不会)"""
        try:
            v = '{{:{n}f}}'.format(n='' if n < 0 else '.{:d}'.format(n)).format(float_x(v)).rstrip('0').rstrip('.')
        except:
            pass
        return v

    @staticmethod
    def format_numeric(x, f1='{:f}'):
        if pd.isna(x):
            return None
        ret = f1.format(float_x(x)).rstrip('0').rstrip('.')
        return '0' if ret == '-0' else ret

    @staticmethod
    def try_format_numeric(x, f1='{:f}', f2='{}'):
        try:
            return f2.format(Checker.format_numeric(x, f1), f2)
        except:
            return x  # if pd.isna(x) else str(x)  # pd.isna(x) x is tuple will raise error

    @staticmethod
    def tuple(x):
        if not x:
            return tuple()
        elif Checker.like_list(x):
            return tuple(x)
        else:
            return (x,)

    @staticmethod
    def list(x):
        if not x:
            return list()
        elif Checker.like_list(x):
            return list(x)
        else:
            return [x]

    @staticmethod
    def zip(x: list, y: list):
        len_x = len(x)
        len_y = len(y)
        return [] if len_x != len_y else zip(x, y)  # zip of list

    @staticmethod
    def attr_dict(conf_dict):
        return AttrDict((k, v) for k, v in conf_dict.items())

    @staticmethod
    def check_tuple(v, type_fn=None, sep=None):
        if not v:  # empty tuple
            ret = tuple()
        elif isinstance(v, str) and sep:  # str split
            ret = tuple(x.strip() for x in v.split(sep))
        elif isinstance(v, Iterable) and not isinstance(v, (str, bytes, tuple)):  # list, tuple etc.
            ret = tuple(v)
        else:
            ret = (v,)

        return tuple(type_fn(x) for x in ret) if type_fn else ret

    @staticmethod
    def check_bool(v):
        return True if str(v)[0].lower() in ("1", "t") else False  # True: ("1", "t"); False: ("0", "f")

    @staticmethod
    def round_down(num, n):
        """float format 是四舍五入的. https://stackoverflow.com/a/37712374/6494418"""
        multiplier = pow(10, n)
        return floor(num * multiplier) / multiplier

    @staticmethod
    def safe_div(up, bottom):
        """reference RATIO: 0/0=0; 1/0=1 OR 0/0=0; 1/0=0"""
        return (up / bottom) if bottom != 0 else (1 if up == 0 else 0)

    @staticmethod
    def format_with_keyd_count(val, c=3):
        vs = str(val).split('.')
        if vs[0] == '0':
            f = "%%.%df" % (len(vs[1]) - len(vs[1].lstrip('0')) + c)
        else:
            f = "%%.%df" % min(max(c - len(vs[0]), 0), len(vs[1]))

        return f % val


class ElapseTime:
    def __init__(self, info):
        self.start_time = time()
        self.info = info

    def end(self):
        elapse = (time() - self.start_time) / 60
        return SU.hf("ElapseTime --- %.2fm --- // %s" % (elapse, self.info))


class TimeMarker:
    def __init__(self, flag='', info=''):
        self._start = time()
        self._c_start = self._start
        self.flag = flag
        self.log_keeper = []

        self.log('Sta', self._c_start, 0, info)

    def __str__(self):
        return " ".join([str(x) for x in self.log_keeper])

    def cut(self, info=""):
        cur_time = time()
        run_time = cur_time - self._c_start
        self._c_start = cur_time
        self.log('Cut', self._c_start, run_time, info)
        return run_time

    def end(self, info='', e=0):
        cur_time = time()
        run_time = cur_time - self._start  # full runtime
        self._c_start = cur_time
        self.log('End', self._c_start, run_time, info)
        if e > 0:
            PF.exit(e)

        return run_time

    def log(self, fn, cur_time, run_time, info):
        ts = DateTime.datetime(cur_time, f='p')
        info_p = ' -- ' + str(info) if info else str(info)
        token = fn[0] if not info else f"{fn[0]}.{info}"
        log_keeper_val = f"{token}:{run_time:.2f}"

        if fn == 'Cut':
            self.log_keeper.append(log_keeper_val)
        elif fn == 'End':
            self.log_keeper.insert(0, log_keeper_val)
            info_p = f'{info_p} -- {str(self)}'

        msg = f"TimeMarker( {self.flag} ).{fn}: {ts}({run_time:.2f}s){info_p}"
        PF.p(msg, layer_back=2)


class PickleUtils:
    @staticmethod
    def load_pickle(file_path):
        with open(file_path, 'rb') as f:
            ret = pickle.load(f)
            PF.p('PickleUitls: load from `%s` success!' % file_path)
            return ret

    @staticmethod
    def dump_pickle(data, file_path):
        makedirs(dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            PF.p('PickleUitls: save to `%s` success!' % file_path)

    @staticmethod
    def load_exist_pickle(file_path, remake=False):
        if exists(file_path) and not remake:
            return PickleUtils.load_pickle(file_path)
        else:
            return None


class MultiProcess:
    """ refs:
    http://blog.adeel.io/2016/11/06/parallelize-pandas-map-or-apply/
    http://www.racketracer.com/2016/07/06/pandas-in-parallel/
    https://stackoverflow.com/a/716482/6494418
    https://stackoverflow.com/a/27027632/6494418
    """

    @staticmethod
    def _get_process_num_core(num_core=None):
        """get process core count.
        float value: make frac of available core, frac * max_core, range (0, 1);
        negative int value: available core to minus, range[0, 1-max_core];
        positive int value: core count;
        otherwise: available core minus 1.
        """
        cpu_count = multiprocessing.cpu_count()
        if num_core is None:
            num_core = multiprocessing.cpu_count() - 1
        elif isinstance(num_core, float):
            num_core = int(cpu_count * num_core)
        elif num_core <= 0:
            num_core = cpu_count + num_core

        if num_core > cpu_count or num_core < 1:
            raise ValueError(
                "MultiProcess core count error! available [1, %d], "
                "but get %s." % (cpu_count, num_core))

        return num_core

    @staticmethod
    def _map_pieces(func, pieces, *args, **kwargs):
        return [func(x, *args, **kwargs) for x in pieces]

    @staticmethod
    def map(func, data_list, num_core=None, single=False, tqdm=True, *args, **kwargs):
        if single:  # single core test
            PF.p("[MultiProcess] single test!")
            if tqdm: data_list = Tqdm(data_list)
            return MultiProcess._map_pieces(func, data_list, *args, **kwargs)

        num_core = MultiProcess._get_process_num_core(num_core)

        PF.p("[MultiProcess] use process core: %d" % num_core)
        data_list = np.array_split(data_list, num_core)

        # add tqdm for tail data block
        if tqdm: data_list = [x if i != num_core - 1 else Tqdm(x)
                              for i, x in enumerate(data_list)]

        with multiprocessing.Pool(num_core) as pool:
            data_list = list(itertools.chain.from_iterable(
                pool.map(functools.partial(
                    MultiProcess._map_pieces, func, *args, **kwargs), data_list)))

        return data_list


class LengthCounter:
    def __init__(self, x, tag="none"):
        if isinstance(x, int):
            self.len_pre = x
        else:
            self.len_pre = len(x)

        self.tag = tag
        PF.p("[LengthCounter] (%s)(%d: init count)" % (tag, self.len_pre))

    def count(self, x, info=""):
        if isinstance(x, int):
            len_tmp = x
        else:
            len_tmp = len(x)

        PF.p('[LengthCounter] (%s)(%d: %d%+d) %s' % (
            self.tag, len_tmp, self.len_pre, len_tmp - self.len_pre, info))
        self.len_pre = len_tmp


class UT:  # UnsortTools
    @staticmethod
    def var2str(v):
        """ variable list to string """
        if ',' not in v:
            v = '\',\''.join(v.split())
        else:
            v = '\',\''.join(''.join(v.split()).split(','))
        v = v.replace('[', '').replace(']', '')
        v = '[\'' + v + '\']'
        PF.p(v)

    @staticmethod
    def drop_duplicates(values):
        seen = set()
        seen_add = seen.add

        return [x for x in values if not (x in seen or seen_add(x))]

    @staticmethod
    def get_index_mapped(values, return_map=False, keep_order=False):
        # drop duplicates and sort
        if keep_order:
            elements = UT.drop_duplicates(values)
        else:
            elements = set(values)

        # get index_map
        values_map = dict((v, k) for k, v in enumerate(elements))

        # replace valeus
        values_mapped = [values_map[i] for i in np.array(values)]

        if return_map:
            return values_mapped, values_map
        else:
            return values_mapped

    @staticmethod
    def move_list_tail_to_head(ls):
        if not isinstance(ls, list):
            ls = list(ls)
        ls.insert(0, ls.pop(-1))
        return ls

    @staticmethod
    def move_val2idx(lst, val, to_idx):
        if not isinstance(lst, list): lst = list(lst)
        if to_idx < 0:  to_idx = len(lst) + to_idx
        lst.pop(lst.index(val))
        lst.insert(to_idx, val)

        return lst

    @staticmethod
    def reverse_dict_map(d_map):
        return {v: k for k, v in d_map.items()}

    @staticmethod
    def get_var_size(var):
        var_size = getsizeof(var)
        for unit in ["b", "k", "m", "g", "t"]:
            if var_size > 1024:
                var_size /= 1024
            else:
                PF.p("[Variable Size] %f%s" % (var_size, unit))
                break

    @staticmethod
    def get_root_var_name(var):
        import inspect
        """https://stackoverflow.com/a/40536047/6494418"""
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0: return names[0]

    @staticmethod
    def linspace(split_length, start=0, end=1):
        return [start + x * (end - start) / split_length for x in range(0, split_length + 1)]

    @staticmethod
    def msg_phone(content="none", oaname="", token="", http_url='ms'):
        """ # http://open-monitor.58corp.com/home
            # https://monitor.58corp.com/help/open-monitor/open_alarm_plaform.html
        """
        if oaname is None or oaname == "": oaname = "wangke09"
        if token is None or token == "": token = "4b447be5ba63c1a37d64b08681bfae75"
        data = {'token': token, 'oaname': oaname}

        if http_url == 'sms':
            http_url = "http://openmsg.monitor.op.58dns.org/channel/sms"
            data['content'] = content
        else:
            http_url = "http://openmsg.monitor.op.58dns.org/channel/meishi"
            data['status'] = 'wktk4py'
            data['detail'] = content

        return post(http_url, data=data).text

    @staticmethod
    def parse_xml_to_dict(xml_path, clean_str_tuple=tuple(), print_len=-1):
        """refs: https://www.programmersought.com/article/13481771573/"""
        import json
        import xml.etree.ElementTree as ET
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import xmltodict

        xml_data = ET.parse(xml_path).getroot()
        xml_string = ET.tostring(xml_data, encoding='utf8', method='xml')

        dict_data = dict(xmltodict.parse(xml_string))
        json_str = json.dumps(dict_data, indent=2, default=lambda o: str(o))
        for c in clean_str_tuple: json_str = json_str.replace(c, '')
        dict_data = json.loads(json_str)

        if print_len != -1:
            if print_len >= 0:
                json_str = json_str[:print_len]
            PF.p("----------------------------------------------------------------")
            PF.p(json_str)
            PF.p("----------------------------------------------------------------")

        return dict_data

    @staticmethod
    def check_match_length(left, right, name_left="left", name_right="right", info="length not match"):
        len_left, len_right = len(left), len(right)
        info = "[ERROR] %s (%s:%s <> %s:%s)!" % (info, name_left, len_left, name_right, len_right)
        assert len_left == len_right, info

    @staticmethod
    def get_cpu_count():
        return multiprocessing.cpu_count()

    @staticmethod
    def append_id_to_filename(filename):
        """How to add an id to filename before extension?
        https://stackoverflow.com/a/37488134/6494418"""
        from random import choice
        from string import ascii_uppercase, digits

        def generate_id(size=7, chars=ascii_uppercase + digits):
            return ''.join(choice(chars) for _ in range(size))

        name, ext = splitext(filename)
        return "{name}_{uid}{ext}".format(name=name, uid=generate_id(), ext=ext)


class Tqdm:
    def __init__(self, iterable, num_marker=10):
        self.iterable = iterable
        self.len = len(self.iterable)
        self.num_marker = num_marker
        self.num_gap = int(self.len / self.num_marker)
        # time
        self.start_time = time()
        self.cut_time = self.start_time
        # PF.p format
        len_1, len_2 = len(str(self.num_marker)), len(str(self.len))
        self.format_print = "[tqdm] (%{}d/{} | %{}d/{}) %fs".format(
            len_1, self.num_marker, len_2, self.len)

    def cut(self):
        current = time()
        run_time = current - self.cut_time
        self.cut_time = current
        return run_time

    def __iter__(self):
        n = 0
        for obj in self.iterable:
            n += 1
            if n % self.num_gap == 0:
                PF.p(self.format_print % (n // self.num_gap, n, self.cut()))
            yield obj
        PF.p("[tqdm] end! runtime: %fs" % (time() - self.start_time))


class Repo:
    @staticmethod
    def update_repo(repo_git, repo_name, data_dir='./', clean=False, **kwargs):
        repo_path = path_join(data_dir, repo_name)
        if exists(repo_path):
            rmtree(repo_path)
        if not clean:
            Bash.sh("cd %s || exit 1 && git clone %s" % (data_dir, repo_git), **kwargs)


class DateTime:
    @staticmethod
    def datetime(seconds=None, f='%Y%m%d_%H%M%S'):
        """given ts, return fmt datetime. gmtime not correct with localtime!"""
        if f == 's':
            f = '%Y%m%d_%H%M%S'
        elif f == 'p':
            f = '%y-%m-%d %H:%M:%S'

        return strftime(f, localtime(seconds))

    @staticmethod
    def date(date=None, day_delta=0, f='%Y%m%d'):
        """given fmt date, return shift fmt date"""
        if f == 's':
            f = '%y%m%d'
        elif f == 'p':
            f = '%y-%m-%d'
        date = datetime.strptime(date, f) if date else datetime.now()  # 转化为日期对象
        return datetime.strftime(date + timedelta(days=day_delta), f)  # 转化为字符串

    @staticmethod
    def time(seconds=None, f='%H%M%S'):
        """given ts, return fmt time"""
        return strftime(f, localtime(seconds))

    @staticmethod
    def date_index(x, f='%Y%m%d'):
        return int(((mktime(strptime(str(x), f)) / 60 / 60) + 8) / 24)

    @staticmethod
    def hour_index(x, f='%Y%m%d%H'):
        return int((mktime(strptime(str(x), f)) / 60 / 60) + 8)

    @staticmethod
    def date2week(date, f='%Y%m%d', both=True):
        try:
            return strftime('%y%m%d-%u' if both else '%u', strptime(SU.strn(date), f))
        except Exception as e:
            return "ERR:" + str(e)

    @staticmethod
    def current_milli_time():
        return floor(time() * 1000)

    @staticmethod
    def check_time(given_time, duration=1.0):
        current_time = DateTime.current_milli_time()
        interval = floor(float_x(current_time - given_time) / 60000)
        if interval > 0 and interval > duration:  # 60000=1min
            PF.exit('[current_time(%s) - check_time(%s) = %s] > %s'
                    % (current_time, given_time, interval, duration))


class Bash:
    @staticmethod
    def sh(cmd, log_file=None, use_sp=False, failed_exit=True):
        run_time = strftime("%y%m%d_%H%M%S", localtime())
        if log_file and isinstance(log_file, str):
            if log_file == "":
                log_file = "./log/%s_bash_sh.log" % RUN_TIME  # auto log_file
            cmd = "%s 2>&1 | tee -a '%s'" % (cmd, log_file)
        else:
            cmd = "%s 2>&1" % cmd

        if 'hadoop fs ' in cmd:
            cmd = cmd.replace(' 2>&1', " 2>'/tmp/shell_wk.err'")

        PF.p("[Bash.sh] [%s] $ %s" % (run_time, Trans.trans_format(cmd)))
        if use_sp:  # use subprocess
            try:
                result = sp_check_output(cmd, shell=True, stderr=SP_STDOUT)
                exit_code, info = 0, result.decode("utf-8")
            except Exception as e:
                exit_code, info = 1, str(e)
        else:  # use os.system
            # os.system是无法记录stdout的(系统调用的输出), 只能打印在终端上
            # 如果需要记录stdout, 只有调用subprocess, 或者在调用python时加 | tee -a "log_file.log"
            exit_code = system(cmd)
            info = exit_code

        sof = "SUCCESS" if exit_code == 0 else "FAILED"
        PF.p("[Bash.sh] [%s] %s(%s)!" % (run_time, sof, info))
        if failed_exit and exit_code != 0:
            raise BrokenPipeError(cmd)  # 退出运行
        return exit_code  # 0: success; else failded

    @staticmethod
    def sql(sql, **kwargs):
        Bash.sh("""hive -e \"%s\"""" % sql, **kwargs)

    @staticmethod
    def sql_file(sql, **kwargs):
        Bash.sh("""hive -f \"%s\"""" % sql, **kwargs)

    @staticmethod
    def text_file_from_hdfs(file_path, out_file, **kwargs):
        """获取执行结果, 基于结果计算各指标. return 0: 执行成功"""
        return Bash.sh("hadoop fs -text {PATH}/* > {OUT}".format(PATH=file_path, OUT=out_file), **kwargs)

    @staticmethod
    def get_clipboard():
        """Return clipboard contents as Unicode.
        from pandas.io.clipboard import clipboard_get"""
        return check_output(['pbpaste']).decode('utf-8')

    @staticmethod
    def write_clipboard(text):
        """https://stackoverflow.com/a/25802742/6494418"""
        Popen(['pbcopy'], stdin=PIPE).communicate(text.encode('utf-8'))

    @staticmethod
    def sh_(cmd):
        PF.p("[Bash.sh] $ %s" % Trans.trans_format(cmd))
        system(cmd)


class File:
    @staticmethod
    def read(path):
        with open(path, "r") as f:
            p = f.read()
        return p

    @staticmethod
    def write(text, path):
        makedirs(dirname(path), exist_ok=True)
        PF.p('[File] [write text to] %s' % path)
        with open(path, "w") as f:
            f.write(text)

    @staticmethod
    def read_text_file(path):
        p = []
        with open(path, 'r') as f:
            for line in f:
                p.append(line)
        return p

    @staticmethod
    def write_text_file(content, path):
        with open(path, 'w') as f:
            f.write(content)

    @staticmethod
    def read_table(path, sep=None, min_len=1, ignore_start=("#",)):
        from io import open
        if isinstance(ignore_start, str):
            ignore_start = (ignore_start,)
        ignore_start = ("", "\n") + ignore_start
        with open(path, "r", encoding="utf-8") as f:
            p = [x.strip().split(sep) for x in f]
        return [x for x in p if len(x) >= min_len and "".join(x)[0] not in ignore_start]

    @staticmethod
    def path_head_flag(path, flag):
        return path_join(dirname(path), '%s_%s' % (flag, basename(path)))

    @staticmethod
    def path_tail_flag(path, flag):
        idx_dot = path.rindex('.')
        return '%s_%s.%s' % (path[:idx_dot], flag, path[idx_dot + 1:])

    @staticmethod
    def path_head_time(path, time_=None):
        time_ = time_ if time_ else DateTime.datetime()
        return File.path_head_flag(path, time_)

    @staticmethod
    def path_tail_time(path, time_=None):
        time_ = time_ if time_ else DateTime.datetime()
        return File.path_tail_flag(path, time_)

    @staticmethod
    def drop_duplicated_line(x):
        try:
            p = []
            with open(x, 'r') as f:
                for line in f:
                    p.append(line.strip())
            p = '\n'.join(list(OrderedDict.fromkeys(p)))
            with open(x, 'w') as f:
                f.write(p)
        except:
            pass

    @staticmethod
    def create_if_not_exist_dir(path):
        """https://stackoverflow.com/a/12517490/6494418"""
        makedirs(path.dirname(path), exist_ok=True)

    @staticmethod
    def print_list_dir(dir_):
        from os import sep
        dir_abs = abspath(dir_)
        info = ('[PRINT LIST DIR] %s' % dir_) + (
            "" if dir_abs == dir_ else '\n[PRINT LIST DIR abs] %s' % dir_abs)
        print('%s\n%s\n%s' % (SEP_LINE_64, info, SEP_LINE_64))
        for root, dirs, files in os_walk(dir_):
            level = root.replace(dir_, '').count(sep)
            indent = '| ' * level
            print('{}{} \\'.format(indent, basename(root)))
            sub_indent = '| ' * (level + 1)
            for f in sorted(files):
                print('{}{}'.format(sub_indent, f))
        print('%s\n%s\n%s' % (SEP_LINE_64, info, SEP_LINE_64) + '\n')

    @staticmethod
    def get_root_dir(file, indent):
        return abspath(('%s/' % dirname(realpath(file))) + ('../' * indent))

    @staticmethod
    def add_sys_path(file, indent):
        sys_path.append(File.get_root_dir(file, indent))  # 程序入口脚本依赖路径

    @staticmethod
    def get_if_idx_path(path):
        """如果路径存在, 添加idx后缀"""
        base, ext = path_splitext(path)
        for i in range(999):
            dup_flag = '' if i == 0 else "%02d" % i
            path_new = base + dup_flag + ext
            if not exists(path_new):
                return path_new
        raise ValueError(path)

    @staticmethod
    def copy_file(src, dst, test=False):
        PF.p("[copy_file] copy(%s) to(%s)" % (src, dst))
        if test: return
        dir_ = dirname(dst)
        if not exists(dir_): makedirs(dir_, exist_ok=True)
        shutil_copy2(src, dst)  # copy2 to keep old meta info

    @staticmethod
    def move_file(src, dst, test=False):
        PF.p("[move_file] copy(%s) to(%s)" % (src, dst))
        if test: return
        dir_ = dirname(dst)
        if not exists(dir_): makedirs(dir_, exist_ok=True)
        shutil_move(src, dst)  # copy2 to keep old meta info

    @staticmethod
    def load_properties(filepath, sep='=', comment_char='#'):
        """Read the file passed as parameter as a properties file.
        refs: https://stackoverflow.com/a/31852401/6494418
        """
        props = {}
        with open(filepath, 'r') as f:
            for line in f:
                x = line.strip()
                if x and not x.startswith(comment_char):
                    sep_idx = x.find(sep)
                    if sep_idx != -1:
                        k = x[:sep_idx]
                        v = x[sep_idx + 1:]
                        if k and not k.startswith('__'):  # __ for comment not work
                            props[k] = v
        return props


class SafeDict(dict):
    def __missing__(self, key):
        """'{A_STRING}'.format_map(SafeDict(
            DATE_=conf.date,
        ))"""
        return '{' + key + '}'


class AttrDict(dict):
    """https://stackoverflow.com/a/39375731/6494418"""

    __ATTR_TUPLE__ = ('_NAME_',)

    def __init__(self, *args, **kwargs):
        if args == (None,): args = tuple()
        super().__init__(*args, **kwargs)
        # self.__dict__ = self  # get as attr with one layer
        # refs: https://stackoverflow.com/a/14620633/6494418 https://stackoverflow.com/a/14620633/6494418
        self._NAME_ = 'AttrDict'  # 变量不能用双下划线开头, 双下划线开头会替换为类名

    def __getitem__(self, key, default=None):
        """可以通过[.属性]的方法获取item https://stackoverflow.com/a/1639632/6494418"""
        if key not in self:
            return default
        # call the super method to prevent recursion call
        value = super().__getitem__(key)
        if type(value) == dict:  # isinstance include subclass
            value = AttrDict(value)
            self[key] = value
        return value

    def __getattr__(self, key, default=None):
        return self.__getitem__(key, default)

    def get(self, key, default=None):
        return self.__getitem__(key, default)

    def __setitem__(self, *args, **kwargs):  # real signature unknown
        super().__setitem__(*args, **kwargs)

    def __setattr__(self, key, value):
        # diff dict key and dict attr
        if key in AttrDict.__ATTR_TUPLE__:
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def str(self, f=True):
        """https://stackoverflow.com/a/3314411/6494418"""
        return dumps(self, sort_keys=False, indent=2, default=lambda o: str(o)) if f else super().__str__()

    def __getstate__(self):
        """multiprocess can not pickle: https://stackoverflow.com/a/2050357/6494418"""
        return self.__dict__

    def __setstate__(self, d):
        """multiprocess can not pickle: https://stackoverflow.com/a/2050357/6494418"""
        self.__dict__.update(d)

    def set_name(self, name):
        self._NAME_ = name
        return self

    def get_name(self):
        return self._NAME_

    def gi(self, key, default=None, info=None):
        """get value with info"""
        info = '' if info is None else ' -- %s' % info
        if key in self:
            flag, value = 'S', self[key]
        elif default is None:
            raise ValueError('[%s] key(%s) NOT FOUND in dict%s' % (self._NAME_, key, info))
        else:
            flag, value = 'D', default

        PF.p('[AttrDict][%s] %s[%s]=%s%s' % (flag, self._NAME_, key, value, info))
        return value


class Yaml:
    """refs:
    https://stackoverflow.com/a/1774043/6494418
    https://stackoverflow.com/a/12471272/6494418"""

    @staticmethod
    def load(file):
        with open(file, 'r') as stream:
            try:
                result = load(stream)
            except YAMLError as exc:
                raise YAMLError(exc)

        return AttrDict(result) if result else AttrDict()

    @staticmethod
    def dump(data, file):
        with open(file, 'w') as outfile:
            dump(data, outfile, sort_keys=False)

    @staticmethod
    def base_config(conf_dict, base_config_key='base_config'):
        """base_config for yaml"""
        for k, v in conf_dict.items():
            if base_config_key in v:
                conf_dict[k] = deepcopy(conf_dict[v.base_config])
                conf_dict[k].update(v)

        return conf_dict


class ReDuplicated:
    """https://stackoverflow.com/a/46332739/6494418"""

    def __init__(self, sep='.'):
        self.d = dict()
        self.sep = sep

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s%s%d" % (x, self.sep, self.d[x])

    @staticmethod
    def list(ll, sep="."):
        """df.columns = ReDuplicated.list(df.columns)"""
        rd = ReDuplicated(sep=sep)
        return [rd(x) for x in ll]

    @staticmethod
    def list2(ll, sep="."):
        """with inner list df.columns = ReDuplicated.list(df.columns)"""
        rd = ReDuplicated(sep=sep)
        return [tuple(rd(xx) for xx in x) if isinstance(x, (list, tuple)) else rd(x) for x in ll]

    @staticmethod
    def clean(ll):
        """https://stackoverflow.com/a/7961425/6494418"""
        return list(dict.fromkeys(ll))


class SU:
    """String utils"""

    @staticmethod
    def dollar_replace_with_dict(string_, repalcer_dict):
        """replace with ${var}"""
        for k, v in repalcer_dict.items():
            string_ = string_.replace('${%s}' % k, v)
        return string_

    @staticmethod
    def replace_with_dict(string_, repalcer_dict):
        for k, v in repalcer_dict.items():
            if not pd.isna(v):
                string_ = string_.replace(str(k), str(v))
        return string_

    @staticmethod
    def hf(s='', le=80, fi="-", pre='\n', f='[ %s ]'):
        if s and f: s = f % s
        return "{pre}{a}{b}{a}{c}".format(a=(fi * int((le - len(s)) / 2)), b=s, c=(fi * (len(s) % 2)), pre=pre)

    @staticmethod
    def java_map_to_json(s):
        """java map print to json"""
        s = s.replace(', ', '@@@').replace(',', '@@').replace('@@@', '", "').replace('@@', ',')
        s = s.replace('=', '": "').replace('{', '{"').replace('}', '"}').replace('"}"}', '"}}').replace('"{', '{')
        return s

    @staticmethod
    def dumps(obj, indent=None):
        return dumps(obj, indent=indent, separators=(',', ':'), ensure_ascii=False, default=lambda o: str(o))

    @staticmethod
    def ss(x=None, sep=None, includes=None, excludes=None):
        if x:
            return [b for b in [a.strip() for a in x.split(sep)]
                    if b and not b.startswith('__')
                    and (includes is None or b in includes)
                    and (excludes is None or b not in excludes)]
        else:
            return []

    @staticmethod
    def strn(x):
        x = str(x)
        return x if not x.endswith('.0') else x[:-2]

    @staticmethod
    def get_paired_kv(x, sep='='):
        sep_idx = x.find(sep)
        if sep_idx != -1:  # 包含`sep`
            return x[:sep_idx], x[sep_idx + len(sep):]
        return x, None

    @staticmethod
    def get_paired_kv_dict(lst, sep='='):
        lst = lst.split() if isinstance(lst, str) else lst
        lst_n_sep_idx = [(x, x.find(sep)) for x in lst]
        return dict((x[:sep_idx], x[sep_idx + 1:]) for (x, sep_idx) in lst_n_sep_idx if sep_idx != -1)

    @staticmethod
    def clean_chars(s, bad_chars="""(){}<>"'"""):
        """translate better than replace
            https://stackoverflow.com/a/3900077/6494418
            https://stackoverflow.com/a/3939381/6494418
        """
        return s.translate({ord(i): None for i in bad_chars})

    @staticmethod
    def format_table(rows, transpose=False):
        """https://stackoverflow.com/a/12065663/6494418"""
        if transpose:
            """https://stackoverflow.com/a/6473724/6494418"""
            rows = list(map(list, itertools.zip_longest(*rows, fillvalue='')))
        widths = [max(map(len, map(str, col))) for col in zip(*rows)]
        p = []
        for row in rows:
            p.append('  '.join((val.ljust(width) for val, width in zip(row, widths))))
        return '\n'.join(p)


class Args:
    CODE_TRUE = ('1', 't')

    @staticmethod
    def check_bool(value):
        if value:
            return str(value)[0] in Args.CODE_TRUE
        return False

    @staticmethod
    def get_paired_args(args_list=None, sep='='):
        if args_list is None:
            args_list = argv[1:]
        args_dict = AttrDict().set_name('paired_args')
        for x in args_list:
            k, v = SU.get_paired_kv(x, sep=sep)
            if k and '/' not in k and v:  # ignore path path/dt=xxx as args to replace
                args_dict[k] = v

        PF.print_dict(args_dict, title='Args.get_paired_args()')
        return args_dict

    @staticmethod
    def local_debug_init(init_argv="""global_args=whoami:c"""):
        import sys  # 这里直接使用对argv赋值需要全局变量
        if len(sys.argv) == 1:  # local debug
            sys.argv += ['--%s' % x.strip() for x in init_argv.split(";")]
            PF.print_list(sys.argv, title="input argv")

    @staticmethod
    def parse_global_args(args):
        if args.global_args is None:
            temp = dict()
        else:
            temp = [x.split(":") for x in str(args.global_args).split(",")]  # (:,)
        args.global_args = AttrDict(temp).set_name('global_args')

    @staticmethod
    def get_parsed_args(parser, global_args=False):
        """args = Args.get_parsed_args(parser)"""
        args, unknown_args = parser.parse_known_args()
        args = AttrDict(vars(args))
        if global_args:
            Args.parse_global_args(args)
        PF.print_dict(args, 'known args')
        PF.print_list((unknown_args if unknown_args else []), "unknown args")
        return args


class REG:
    s_test = "18 18.2 118.2 118.20 118.202 18 -18.2-  -18.2 aaa 5.18 18. .18 .18"
    PUNCTUATION = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~»–—‘’‛“”„‟…‧、。〃《》「」『』【】〔〕〖〗〘〙〚〛
    〜〝〞〟〰〾〿﹏！＂＃＄％＆＇（）＊＋，－／：；＜＝＞？＠［＼］＾＿｀｛｜｝～｟｠｡｢｣､￥"""

    @staticmethod
    def chs(s, repl=' '):
        """ 只保留中文字符, 使用空格分割 """
        # s = repl.join(sub('[^\u4e00-\u9fa5]', repl, s).split())
        return repl.join(split('[^\u4e00-\u9fa5]', s))

    @staticmethod
    def eng(s, repl=' '):
        """ 只保留英文字符, 使用空格分割 """
        return repl.join(split('[^a-zA-Z]', s))

    @staticmethod
    def num(s, repl=' '):
        """ 只保留数字字符, 使用空格分割 """
        return repl.join(split('[^0-9]', s))

    @staticmethod
    def html(s):
        """ 去除html标签 """
        s = sub('<.*?>', ' ', s)
        s = sub('&([0-9a-zA-Z]+);', '', s)
        s = ' '.join(s.split())

        return s

    @staticmethod
    def find_all_num(s):
        """return a list extract all numeric, include int and float."""
        return [float_x(x) if '.' in x else int(x) for x in findall(r'-?\d+\.?\d*', s)]

    @staticmethod
    def word(s, replacer=' '):
        """keep eng, chs, dig, and space.
        https://github.com/Shuang0420/Shuang0420.github.io/wiki/python-清理数据, 仅保留字母, 数字, 中文"""
        return sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", replacer, s)

    @staticmethod
    def is_only_chs(x):
        """判断x里面是否全为中文"""
        return search(r"[a-zA-Z0-9]", x) is None

    @staticmethod
    def parentheses(x, content=False):
        """get content of (*)."""
        idx_l, idx_r = x.find('('), x.rfind(')')
        if idx_l != -1 and idx_r != -1:
            x = x[idx_l + 1:idx_r] if content else x[:idx_l] + x[idx_r + 1:]
        return x

    @staticmethod
    def check_clean(s):
        s2 = REG.clean_punc(s)
        if s2 != s:
            raise ValueError('str(%s) contain bad char(%s)' % (s, set(s) - set(s2)))
        return s2

    @staticmethod
    def replace_whole_word(string, _old, _new, ignore=""):
        pattern = r"(?<!{b})\b(?!{a}{b})({a})\b".format(a=_old, b=ignore) if ignore else r"\b{a}\b".format(a=_old)
        return sub(pattern, _new, string)

    @staticmethod
    def clean_punc(s, p='_'):
        """https://stackoverflow.com/a/38799620/6494418 with performance"""
        return sub(r'[^\w]+', p, s).strip(p)

    @staticmethod
    def to_snake_case(name, sep='_'):
        """https://stackoverflow.com/a/1176023/6494418"""
        name = sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        if sep != '_':
            name = sub('__([A-Z])', r'_\1', name)
        name = sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.lower()

    @staticmethod
    def snake_case_to_pascal_case(name):
        name = ''.join(word.title() for word in name.split('_'))
        return name

    @staticmethod
    def get_img_url_text(html):
        return findall(r'!\[(.*?)]\((.*?)\)', str(html))

    @staticmethod
    def get_html_url_text(html):
        return findall(r'href=[\'"]?([^\'" >]+).*?>(.*)<', str(html))

    @staticmethod
    def get_startswith_http(html):
        return findall(r'(https?://.*?)[\'"]', str(html))  # ? 是可能存在, .* 是任何字符


class Trans:
    REP_ENG = """\
，=, 
。=. 
：=: 
；=; 
！=! 
？=? 
、=, 
【=[
】=]
（=(
）=)
％=%
＃=#
＠=@
＆=&
—=-
“="
”="
《=<
》=>
"""  # 注意部分等号右边的符号有一个空格
    REP_ENG = [x.split("=") for x in [x for x in REP_ENG.split('\n')] if x and not x.startswith("#")]
    REP_ENG = dict([(ord(x[0]), x[1]) for x in REP_ENG if len(x[0]) == 1])

    @staticmethod
    def trans_line(s):
        return ' '.join(s.translate(Trans.REP_ENG).split())

    @staticmethod
    def trans_line_indent(s):
        left_space = len(s) - len(s.lstrip())
        return (' ' * left_space) + Trans.trans_line(s)

    @staticmethod
    def trans_content(content):
        result = []
        for line in content.split("\n"):
            result.append(Trans.trans_line_indent(line))

        return "\n".join(result)

    @staticmethod
    def trans_format(content):
        """
        REP_FMT = '''\t=\\t  \n=\\n  \001=\\001  \002=\\002  \003=\\003'''
        REP_FMT = [x.split("=") for x in [x for x in REP_FMT.split('  ')] if x and not x.startswith("#")]
        REP_FMT = dict([(ord(x[0]), x[1]) for x in REP_FMT if len(x[0]) == 1])
        return content.translate(Trans.REP_FMT)
        1. r"xxx"标识raw_string, 注意, 这里的raw_string是你在屏幕上看到的是什么, 打印出来的就是什么
        2. 被转义的\t在计算机中就是tab, 被转义的\n在计算机中就是newline, 他们是无法区分的?(是一个符号, 所以替换\n, newline也会被替换)
        3. 所以用translate不能实现该功能, 所有的\n, newline都会被替换为\\n(还是只有用加引号的replace区分\n和newline, 注意: "newline"和"\n"仍然是无法区分的)
        https://stackoverflow.com/questions/2428117/casting-raw-strings-python
        https://stackoverflow.com/questions/18707338/print-raw-string-from-variable-not-getting-the-answers
        上述方法试过都无效
        """
        return (content.replace("'\n'", "'\\n'").replace('"\n"', '"\\n"')
                .replace("'\t'", "'\\t'").replace('"\t"', '"\\t"'))


class JU:
    @staticmethod
    def load_multi_line_json_file(path):
        with open(path, 'r') as jf:
            return load(jf)

    @staticmethod
    def save_multi_line_json_file(j, path):
        from json import dump
        with open(path, 'w', encoding='utf8') as jf:
            dump(j, jf, indent=2, ensure_ascii=False, sort_keys=False)

    @staticmethod
    def flatten_json(nested_json, exclude=[], sep='.'):
        """https://stackoverflow.com/a/57334325/6494418"""
        if isinstance(nested_json, str):
            nested_json = loads(nested_json)
        out = {}

        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    if a not in exclude:
                        flatten(x[a], name + a + sep)
            elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + sep)
                    i += 1
            else:
                out[name[:-1]] = x

        flatten(nested_json)
        return out


def t2():
    content = """蓬莱一品民俗家庭公寓
    蓬莱金萍渔家乐
    蓬莱凯莱宾馆
    速8酒店(蓬莱蓬莱阁登州路店)
    蓬莱仙境姐姐家家庭驿站
    蓬莱秋实家庭公寓
    烟台阳光味道特色民宿
    烟台琦秀度假公寓(沿河街分店)
    烟台无限风光在木石99公寓(观音苑分店)
    烟台仙境爱侣行汐•遇公寓(海洋极地世界分店)
    """
    content = content.split()

    for i in content:
        PF.p(REG.parentheses(i, content=True))


def t1():
    ts = TimeMarker()
    ts.cut("hello")
    ts.end()
    PF.PRT = True
    PF.print_list([1, 2, 3, 4])
    PF.init('2', string_io=True)
    PF.print_dict({1: 1, 2: 2})
    PF.print_block('123', '456')
    msg = PF.print_obj({1: 1, 2: 2})
    print(123, msg)
    print(345, PF.get_string())
    PF.print_dict(Logger._LOG_DICT_)


def t3():
    attr_dict = AttrDict(None)
    a = """1
2	
3
1\t2\n3\t"""
    print(a.encode('unicode_escape').decode())
    print(repr(a))
    print(r"{}".format(a))
    print(Trans.trans_format("123\t12'3;'4]\n456"))
    Bash.sh('sh /Users/wangke/working/zhaopin_data_offline/bin/bash_test_.sh')


def t4(a, b):
    try:
        float_x('m')
    except Exception as e:
        PF.print_stack(e=e)


def t5():
    # t3()
    # UT.msg_phone('meishi', http_url='ms')
    # t4(1, 2)
    print(SU.clean_chars("()真的是,这个吗''""'"))
    # Args.get_paired_args("1 2 3 4=5".split())
    PF.p('hello')
    tm = TimeMarker('train')
    tm.cut('hello')
    tm.end('running end')

    # PF.p(SU.format_table([['a', 'b', 'c'], ['aaaaa', 'b', 'c'], ['a', 'bbbbbbbbbb', 'c']], transpose=True), nl=3)
    # PF.p(SU.format_table([['a', 'b', 'c'], ['aaaaa', 'b', 'c'], ['a', 'bbbbbbbbbb', 'c']]), nl=3)


if __name__ == '__main__':
    # do Not assign variable here, will hint `Shadows name 'variable' from outer scope `
    t5()
