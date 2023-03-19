import re
from Transfer_Tools.convertor_tool_lib.Tool_Base import ToolBaseClass


class SignFix(ToolBaseClass):
    """
    将代码中的一些符号替换为 python符号。
    1. 删除句尾的分号（考虑到有些特殊的分号在句中且可能在字符串内，所以暂时不考虑。
    2. * 替换 if 语句中的 ~ 为 not，&& 为 and，|| 为 or
    3.
    """

    def __init__(self):
        super().__init__()
        self.name = "SignFix"

    def process(self, lines) -> str:
        # lines = str(lines)
        for i in range(len(lines)):
            if len(lines[i].strip()) > 0:
                if lines[i].strip()[0] == '#':
                    continue
            else:
                continue

            content = lines[i].rstrip()
            comment = None
            # 如果句尾有注释，则先分离注释。
            if len(content.split("#", 1)) > 1:
                comment = content.split("#", 1)[1]
                content = content.split("#", 1)[0]

            # 清理行尾分号
            content = content.rstrip().strip(";")

            # TODO 这里还要加上其他功能，例如替换 or, and, not 等

            # 最后替换原句。
            lines[i] = content + "\n" if comment is None else content + "  # " + comment + "\n"

        return lines
