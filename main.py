import sys
import socket
from socket import SOL_SOCKET, SO_REUSEADDR
from PyQt5.QtWidgets import QMainWindow, QApplication, QStyleFactory
from Panel.MainPanel import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.start_server_btn.clicked.connect(self.start_server)
        self.client_connect_btn.clicked.connect(self.connect_server)
        if not self.read_config():
            # 删除相应的库
            self.Data_source_selection.takeItem(5)
            self.Data_source_pages.removeWidget(self.Siemens3T_P)

    def connect_server(self):
        sk = socket.socket()
        sk.connect(('127.0.0.1', 8898))
        sk.send(b'hello!')
        ret = sk.recv(1024)
        print(ret)
        sk.close()

    def read_config(self):
        config_file = "config/LibLoad.config"
        with open(config_file, mode='r') as conf_file:
            lines = conf_file.readlines()
            for line in lines:
                if line == '>Siemens 3T:\n':
                    return True
            return False

    def start_server(self):
        sk = socket.socket()
        sk.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)  # 就是它，在bind前加
        sk.bind(('127.0.0.1', 8898))  # 把地址绑定到套接字
        sk.listen()  # 监听链接
        conn, addr = sk.accept()  # 接受客户端链接
        ret = conn.recv(1024)  # 接收客户端信息
        print(ret)  # 打印客户端信息
        conn.send(b'hi')  # 向客户端发送信息
        conn.close()  # 关闭客户端套接字
        sk.close()  # 关闭服务器套接字(可选)


if __name__ == '__main__':
    QApplication.setStyle(QStyleFactory.create('Fusion'))
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
