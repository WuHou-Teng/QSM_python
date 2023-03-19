import re
from Transfer_Tools.convertor_tool_lib.Tool_Base import ToolBaseClass


class IndexFix(ToolBaseClass):
    def __init__(self):
        super().__init__()
        self.name = 'IndexFix'

    def get_name(self):
        return self.name

    # 工具1，对自动将index替换成成python的
    def process(self, lines):

        # string = 'tmp=[tp(:,3:7) tp(:,1:2)];'
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

            results = re.findall(r'\d*:\d*', content)

            if len(results) != 0:
                for part in results:
                    if len(part) <= 1:
                        continue
                    else:
                        [num1, num2] = part.split(':')
                        if len(num1) > 0:
                            num1 = str(int(num1) - 1)
                        # python的索引不包括结尾数字，但是matlab包括，所以此处的转换正好不需要改动第二个数字。
                        # if len(num2) > 0:
                        #     num2 = str(int(num2) + 1)
                        new_index = num1 + ':' + num2
                        string_list = content.split(part, 1)
                        string_list.insert(1, new_index)
                        content = ''.join(string_list)
                # 最后替换原句。
                lines[i] = content + "\n" if comment is None else content + "  # " + comment + "\n"

        return lines
    # def replace_index(file_address):
