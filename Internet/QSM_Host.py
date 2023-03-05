import socket

# 创建socket对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 获取主机名和端口号
host = socket.gethostname()
print(host)
port = 12345

# 绑定socket到指定的主机名和端口号
server_socket.bind((host, port))

# 设置最大连接数
server_socket.listen(1)

# 等待客户端连接
print('等待客户端连接...')
client_socket, client_address = server_socket.accept()
print('连接地址:', client_address)

# 接收客户端发送的数据，并发送响应
while True:
    data = client_socket.recv(1024).decode('utf-8')
    if not data:
        break
    print('收到数据:', data)
    response = '收到数据: ' + data
    client_socket.send(response.encode('utf-8'))

# 关闭连接
client_socket.close()
server_socket.close()
