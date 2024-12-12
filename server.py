#!/usr/bin/env python
# coding=utf8
# ===============================================================================
#   Copyright (C) 2024 www.361way.com site All rights reserved.
#
#   Filename      ：server.py
#   Author        ：yangbk <itybku@139.com>
#   Create Time   ：2024-12-12 14:37
#   Description   ：
# ===============================================================================

import socket

# 服务器配置
SERVER_IP = '0.0.0.0'  # 监听所有可用网络接口
SERVER_PORT = 12345   # 选择一个未被使用的端口

# 创建UDP套接字
def create_udp_server():
    # 使用AF_INET (IPv4) 和 SOCK_DGRAM (UDP)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定IP和端口
    server_socket.bind((SERVER_IP, SERVER_PORT))

    print(f"UDP服务器已启动，监听 {SERVER_IP}:{SERVER_PORT}")

    while True:
        try:
            # 接收数据
            data, client_address = server_socket.recvfrom(1024)  # 缓冲区大小1024字节

            # 解码并打印接收到的数据
            message = data.decode('utf-8')
            print(f"收到来自 {client_address} 的消息: {message}")

            # 可选：发送回复
            response = f"服务器已接收: {message}"
            server_socket.sendto(response.encode('utf-8'), client_address)

        except Exception as e:
            print(f"发生错误: {e}")

# 启动服务器
if __name__ == '__main__':
    create_udp_server()
