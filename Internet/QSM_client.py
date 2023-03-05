import socket

# 创建socket对象
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 获取主机名和端口号
host = socket.gethostname()
print(host)
port = 12345

# 连接到服务器
client_socket.connect((host, port))

# 发送数据
message = input('请输入要发送的数据：')
client_socket.send(message.encode('utf-8'))

# 接收服务器发送的响应
data = client_socket.recv(1024).decode('utf-8')
print('收到响应:', data)

# 关闭连接
client_socket.close()
