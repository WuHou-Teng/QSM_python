import os

from win32ui import CreateFileDialog
from Transfer_Tools.convertor_tool_lib.array_tools.convertor_index_fix import IndexFix
from Transfer_Tools.convertor_tool_lib.array_tools.convertor_mlist_to_np_array import NpArrayTransfer
from Transfer_Tools.convertor_tool_lib.struct_tools.convertor_comments_fix import CommentsFix
from Transfer_Tools.convertor_tool_lib.struct_tools.convertor_sign_fix import SignFix
from Transfer_Tools.convertor_tool_lib.struct_tools.convertor_retract_fix import RetractFix

"""
    优先度
    xlsread 和对应的输出
    所有位于等号左侧的[] （基本都是输出项）
    等号右侧的[] (替换为array)
    size() (np.size())
    索引
    
"""


class Mat2PyConverter(object):
    # 首先，要能输入一个文件名，根据文件名创建一个新的文件，以py结尾
    def __init__(self):
        self.cwd = os.getcwd()

        # E:\work\Interesting_things\机器学习_winter_project\
        # transformer AI\YI\Chapter 2.2-Health index\constr_newdata_transgrid.m
        self.dlg = CreateFileDialog(1)
        # 默认目录
        self.dlg.SetOFNInitialDir(self.cwd)
        # 副本数记录
        self.trail = 0
        # 文件指针
        self.current_file = None
        # 文件名
        self.filename = ''
        # 转换过程中的临时文件名，
        self.reading_filename = ''

    def choose_file(self):
        # 显示对话框
        self.dlg.DoModal()
        # 获取文件名
        self.filename = self.dlg.GetPathName()

    def get_file_and_converting(self):
        self.choose_file()
        self.reading_filename = self.filename
        # 整理注释
        self.open_convert_close(CommentsFix)
        # 整理符号
        self.open_convert_close(SignFix)
        # 修理index
        self.open_convert_close(IndexFix)
        # 将matlab中创建的列表改成array
        self.open_convert_close(NpArrayTransfer)
        # 整理缩进
        self.open_convert_close(RetractFix)

    # TODO 考虑到目前测试，先采用这种固定的形式去逐个调用转换器。
    def open_convert_close(self, process_class):
        """
        调用输入的转换器类，对matlab文件进行一定的转换。
        :param process_class: 转换器类。所有继承ToolBaseClass的类。
        :return:
        """
        with open(self.reading_filename, 'r', encoding='utf-8') as mat_file:
            # 在当前文件夹创建一个.py文件
            address, name = self.get_file_address_name(self.filename)
            tool = process_class()
            self.new_trail(address, name, tool.get_name())  # 这里就更新了下一轮的reading_filename
            lines = mat_file.readlines()
            new_lines = tool.process(lines)
            # 直接将整个文件交给tool，然后将要保存的文件保存下来。
            for line in new_lines:
                self.current_file.write(line)
            # for lines in lines:
            #     # 跳过存粹为注释的行
            #     if len(lines.strip()) > 0:
            #         if lines.strip()[0] == '%':
            #             continue
            #     # 考虑到有些行可能后面跟着注释，所以先吧注释摘掉。
            #     line_no_comment = lines.split('%')[0]
            #     new_line = tool.process(line_no_comment)
            #     # 如果确实行尾有注释
            #     if len(lines.split('%')) > 1:
            #         new_line = new_line + '%' + lines.split('%')[1]
            #     self.current_file.write(new_line)
            # 完成转化，将两个文件都关闭。
            mat_file.close()
            self.current_file.close()

    def get_file_address_name(self, filename=''):
        name = filename.strip().split('\\')[-1].split('.')[0]
        address = '\\'.join(filename.strip().split('\\')[:-1])
        return address, name

    def new_trail(self, address, name, process_name):
        filename = address + '\\' + name + '_trail_' + str(self.trail) + '_' + process_name + '.m'
        try:
            self.current_file = open(filename, 'w')
        except:
            self.current_file.close()
            self.current_file = open(filename, 'w')
        self.trail += 1
        # 更新下一轮要读取的文件名
        self.reading_filename = filename


if __name__ == '__main__':
    converter = Mat2PyConverter()
    converter.get_file_and_converting()

