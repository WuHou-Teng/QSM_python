

class ToolBaseClass(object):

    def __init__(self):
        self.name = ""

    def get_name(self):
        return self.name

    def fit_struct(self, line) -> bool:
        """
        检测该行是否符合转换器需要的结构。
        :return:
        """
        return len(line) > 0

    def process(self, lines) -> str:
        """
        对输入的行加以检测，修改，并返回。
        :param lines:
        :return:
        """
        return lines

