import time


# 定义一个 callback 函数，用于输出迭代信息
class CallBackTool(object):
    def __init__(self):
        self.start_time = time.process_time()
        self.total_time_used = time.process_time() - self.start_time
        self.current_time = time.process_time()
        self.iter_num = 1
        self.part_time = time.process_time() - self.current_time
        self.average_time = self.total_time_used / self.iter_num

    def csg_callback_func(self, x):
        self.update_time()
        print(f"迭代次数:{self.iter_num}, 此次迭代耗时:{self.part_time}秒")
        # print(f"")
        self.iter_num += 1

    def update_time(self, iter_num=1):
        self.part_time = time.process_time() - self.current_time
        self.current_time = time.process_time()
        self.total_time_used = time.process_time() - self.start_time
        self.iter_num = iter_num
        self.average_time = self.total_time_used / self.iter_num

    def print_time(self):
        self.update_time()
        print(f"程序总耗时:{self.total_time_used}秒")
        print(f"阶段耗时:{self.part_time}秒")

    def print_time_line(self, line):
        self.update_time()
        print(f"程序总耗时:{self.total_time_used}秒")
        print(f"阶段耗时:{self.part_time}秒, 运行到程序第{line}行")

    def print_time_iter(self, iters):
        self.update_time()
        print(f"程序总耗时:{self.total_time_used}秒")
        print(f"阶段耗时:{self.part_time}秒, 运行到程序第{iters}次迭代")

    def print_predict_time(self, iters, total_iters):
        self.update_time(iter_num=iters)
        print(f"预计剩余时间:{self.average_time * (total_iters - iters)}秒")
