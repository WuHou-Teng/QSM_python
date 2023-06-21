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
        self.iter_num += 1
        self.update_time(self.iter_num)
        print(f"Iter_num: {self.iter_num}, This iter spent: {self.part_time} seconds")
        print(f"resid:{x}")
        # print(f"")

    def update_time(self, iter_num=1):
        self.part_time = time.process_time() - self.current_time
        self.current_time = time.process_time()
        self.total_time_used = time.process_time() - self.start_time
        self.iter_num = iter_num
        self.average_time = self.total_time_used / self.iter_num

    def print_time(self):
        self.update_time()
        print(f"Total time spent:{self.total_time_used} seconds")
        print(f"Iter time spent:{self.part_time} seconds")

    def print_time_line(self, line):
        self.update_time()
        print(f"Total time spent:{self.total_time_used} seconds")
        print(f"Iter time spent:{self.part_time} seconds, Running to line {line}")

    def print_time_iter(self, iters):
        self.update_time()
        print(f"Total time spent:{self.total_time_used} seconds")
        print(f"Iter time spent:{self.part_time} seconds, Iter_num:{iters}")

    def print_predict_time(self, iters, total_iters):
        self.update_time(iter_num=iters)
        print(f"Time remaining:{self.average_time * (total_iters - iters)} seconds")

    def print_time_function_start(self, function_name):
        self.update_time()
        print(f"Start running function:{function_name}")

    def print_time_function_end(self, function_name):
        self.update_time()
        print(f"Finished running function:{function_name}")
        print(f"Function time spent:{self.part_time} seconds")
