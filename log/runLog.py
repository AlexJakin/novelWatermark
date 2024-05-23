import datetime
import logging
import os

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)

class MyLogging(logging.Logger):
    def __init__(self, name, level = logging.INFO, file = None):

        super().__init__(name, level)

        fmt = "%(asctime)s %(name)s %(levelname)s %(filename)s--%(lineno)dline :%(message)s"
        formatter = logging.Formatter(fmt)

        if file:
            handle2 = logging.FileHandler(file, encoding="utf-8")
            handle2.setFormatter(formatter)
            self.addHandler(handle2)

        else:
            handle1 = logging.StreamHandler()
            handle1.setFormatter(formatter)
            self.addHandler(handle1)

# current_datetime = datetime.datetime.now()
# formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H:%M:%S")
#
#
# logger = MyLogging("mylog",file=root_path+"/out/log/"+formatted_datetime+".log")
#
# if __name__ == '__main__':
#     mlogger = MyLogging("abc")
#     mlogger.info("封装好的日志类，console")
#     logger.info("封装好的日志，文件渠道测试")

