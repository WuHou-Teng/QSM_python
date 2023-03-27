import time


# 定义一个 callback 函数，用于输出迭代信息
class CallBackTool(object):
    def __init__(self):
        self.start_time = time.process_time()
        self.total_time_used = time.process_time() - self.start_time
        self.current_time = time.process_time()
        self.iter_num = 1
        self.part_time = time.process_time() - self.current_time

    def csg_callback_func(self, x):
        self.update_time()
        print(f"迭代次数:{self.iter_num}, 此次迭代耗时:{self.part_time}秒")
        # print(f"")
        self.iter_num += 1

    def update_time(self):
        self.part_time = time.process_time() - self.current_time
        self.current_time = time.process_time()
        self.total_time_used = time.process_time() - self.start_time

    def print_time(self, line):
        self.update_time()
        print(f"程序总耗时:{self.total_time_used}秒")
        print(f"阶段耗时:{self.part_time}秒, 运行到程序第{line}行")
