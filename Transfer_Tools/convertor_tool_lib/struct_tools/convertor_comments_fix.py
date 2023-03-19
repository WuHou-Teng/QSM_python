import re
from Transfer_Tools.convertor_tool_lib.Tool_Base import ToolBaseClass


class CommentsFix(ToolBaseClass):
    """
    将matlab文件中，所有的注释替换为 #
    """

    def __init__(self):
        super().__init__()
        self.name = "CommentsFix"

    def process(self, lines) -> str:
        # lines = str(lines)
        for i in range(len(lines)):
            lines[i] = lines[i].replace("%", "#", 1)

        return lines
