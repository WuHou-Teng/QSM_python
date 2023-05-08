import re
from Transfer_Tools.convertor_tool_lib.Tool_Base import ToolBaseClass


class RetractFix(ToolBaseClass):
    """
    将matlab文件中，所有的注释
    """
    keywords = ["function", "if", "for", "while", "classdef", "properties", "methods", "switch"]
    transit_keywords = ["else", "elseif"]
    switch_keywords = ["case", "otherwise"]
    KEYWORD_END = "end"
    STATE_NORMAL = 0
    STATE_START = 1
    STATE_TRAN = 2
    STATE_END = 3
    STATE_SWITCH = 4
    SWITCH_CHAIN = 0

    def __init__(self):
        super().__init__()
        self.name = "RetractFix"
        self.retract_num = 0

    def process(self, lines) -> str:
        # lines = str(lines)
        for i in range(len(lines)):
            if len(lines[i].strip()) <= 0:
                continue
            content = lines[i].lstrip()
            # print(content, end=None)
            state = self.keyword_check(content)
            # 关于switch语句的逻辑可能还有问题，需要进一步测试
            if state is self.STATE_SWITCH:
                if self.SWITCH_CHAIN == 0:
                    self.SWITCH_CHAIN = 1
                    # 在switch开始的第一个case，多缩进一次。
                    self.retract_num += 1
                state = self.STATE_TRAN
            if state is self.STATE_TRAN or state is self.STATE_END:
                self.retract_num = self.retract_num - 1 if self.retract_num >= 1 else 0
                # 如果原本处于switch语句中，遇到end后再次缩进，标志着switch语句结束
                if state is self.STATE_END and self.SWITCH_CHAIN == 1:
                    self.SWITCH_CHAIN = 0
                    self.retract_num = self.retract_num - 1 if self.retract_num >= 1 else 0
            # print(f"缩进 {self.retract_num} 次")
            lines[i] = "\t" * self.retract_num + content
            # print(lines[i])

            if state is self.STATE_START or state is self.STATE_TRAN:
                self.retract_num += 1
            # elif state is self.STATE_END:
            #     self.retract_num -= 1
        return lines

    def keyword_check(self, string):
        for keyword in self.keywords:
            if string.startswith(keyword):
                return self.STATE_START

        for tranword in self.transit_keywords:
            if string.startswith(tranword):
                return self.STATE_TRAN

        for switchword in self.switch_keywords:
            if string.startswith(switchword):
                return self.STATE_SWITCH

        if string.startswith(self.KEYWORD_END):
            return self.STATE_END

        return self.STATE_NORMAL


