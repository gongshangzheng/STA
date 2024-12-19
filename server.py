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
def udp_server(host='0.0.0.0', port=8080):
    # 创建一个 UDP 套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定到指定的地址和端口
    server_socket.bind((host, port))

    print(f"UDP server is listening on {host}:{port}")

    try:
        while True:
            # 接收数据
            data, addr = server_socket.recvfrom(1024)  # 缓冲区大小为1024字节
            print(f"Received message from {addr}: {data.decode('utf-8')}")

            # 这里可以添加处理数据的逻辑
            # 例如，回复客户端
            # server_socket.sendto(b"Message received", addr)

    except KeyboardInterrupt:
        print("Server is shutting down.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    udp_server()
