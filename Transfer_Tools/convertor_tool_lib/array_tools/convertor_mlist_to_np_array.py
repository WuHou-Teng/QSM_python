import re
from Transfer_Tools.convertor_tool_lib.Tool_Base import ToolBaseClass


class NpArrayTransfer(ToolBaseClass):
    def __init__(self):
        super().__init__()
        self.name = 'NpArrayTransfer'

    def get_name(self):
        return self.name

    def process(self, lines):

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

            # string = 'tmp=[tp(:,3:7) tp(:,1:2)];'
            result = re.findall(r'=\s*\[', content)

            # 这里假设一行只有一个 '=[' 吧

            if len(result) != 0:
                result_line = []
                # 其实这里肯定是只有一个的，不需要遍历，我懒得改了。
                # 考虑到可能存在多行挤在一起用分号分隔的情况，for我就不删了，以后看情况。
                for indexes in result:
                    if len(indexes) <= 1:
                        continue
                    else:
                        # pos = lines.find(indexes)
                        array_str = content.split('=', 1)[1]
                        # print(array_str)
                        # 要先清理一下开头和结尾的空格
                        array_str_clear_outer_sign = array_str.strip().strip(';').strip(']').strip('[')
                        array_str_elements = array_str_clear_outer_sign.rstrip().strip(';').split(';')
                        # print(array_str_clear_outer_sign)
                        # result_line.clear()
                        for element in array_str_elements:
                            sa = element.split(' ')
                            jo = ', '.join(sa)
                            jo = '[' + jo + ']'
                            result_line.append(jo)
                # print(result_line)
                content = content.split('=', 1)[0].strip() + ' = np.array(' + str(result_line).replace('\'', '') + ')'
                # 最后替换原句。
                lines[i] = content + "\n" if comment is None else content + "  # " + comment + "\n"

        return lines
        # return lines.rstrip().strip(';') + '\n'

# string = '[[[RX_sample_sort[0,0];RX_sample_sort[1,0];RX_sample_sort[2,0];RX_sample_sort[3,0];RX_sample_sort[4,0];RX_sample_sort[5,0];RX_sample_sort[6,0]]]]'
# string = '[tp(:,2:7) tp(:,0:2)];'
#
# a = string.strip(';').strip(']').strip('[')
# b = a.split(';')
#
# result = []
#
# for element in b:
#     sa = element.split(' ')
#     jo = ', '.join(sa)
#     jo = '[' + jo + ']'
#     result.append(jo)
#
# print('np.array(' + str(result).replace('\'', '') + ')')


