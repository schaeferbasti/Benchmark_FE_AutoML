import os
import sys

class Logger:

    @staticmethod
    def Log(msg: str):
        print(msg)

    @staticmethod
    def Info(msg: str):
        print(msg)

    @staticmethod
    def Error(msg: str, ex: Exception =None):
        print(msg)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    @staticmethod
    def Warn(msg: str):
        print(f'WARN: {msg}')