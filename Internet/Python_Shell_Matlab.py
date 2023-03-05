
# 1. 使用命令行参数：
# #!/bin/bash
# matlab -nodisplay -nosplash -r "addpath('/path/to/your/matlab/code'); your_matlab_function($arg1,$arg2); exit;"
# 在这个示例中，您需要将
#   "/path/to/your/matlab/code" 替换为您的 Matlab 代码所在的路径，
#   "your_matlab_function" 替换为您要调用的 Matlab 函数名，
#   "$arg1" 和 "$arg2" 替换为您要传递给 Matlab 函数的参数。使用这个脚本，您可以在命令行中运行脚本，将参数传递给 Matlab 函数，并获得输出结果。

# 2. 使用脚本文件
# #!/bin/bash
# matlab -nodisplay -nosplash -r "/path/to/your/matlab/script.m; exit;"
# 在这个示例中，您需要将
#   "/path/to/your/matlab/script.m" 替换为您要执行的 Matlab 脚本文件的完整路径。
#   使用这个脚本，您可以在命令行中运行脚本，并在 Matlab 中执行该脚本文件中的所有代码。
#   您还可以将此脚本与其他 Shell 命令和工具组合使用，以自动化和批量化 Matlab 代码的执行和处理。


# 您可以使用 Python 的 "subprocess" 模块来执行 Shell 命令，并使用标准输入输出和错误流来处理 Matlab 的输入和输出。
import subprocess

# 定义 Matlab 代码的输入参数
arg1 = "value1"
arg2 = "value2"

# 定义 Matlab 代码的 Shell 命令
cmd = "matlab -nodisplay -nosplash -r 'addpath(\'/path/to/your/matlab/code\'); your_matlab_function(\"{}\",\"{}\")'".format(arg1,arg2)

# 执行 Shell 命令，并获取标准输出和错误流
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
out, err = p.communicate()

# 处理 Matlab 代码的输出结果
if out:
    print(out.decode("utf-8"))
if err:
    print(err.decode("utf-8"))


# 在这个示例中，您需要将
# "/path/to/your/matlab/code" 替换为您的 Matlab 代码所在的路径，
# "your_matlab_function" 替换为您要调用的 Matlab 函数名，
# "value1" 和 "value2" 替换为您要传递给 Matlab 函数的参数。
# 使用这个 Python 代码，您可以在 Python 中调用 Matlab 代码，获取其输出结果，并进行进一步处理和分析。

