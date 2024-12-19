#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import json

import math
import time
from dataclasses import dataclass, field
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
import concurrent.futures
# import onnx
# import onnxruntime as ort
from ultralytics import YOLO
import socket
import itertools


# In[ ]:
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.attn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        attn = torch.matmul(query, key.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, value)
        out = self.attn_conv(out)
        return out

def create_conv_block(in_channels, out_channels, kernel_size=3, padding=1):
    """
    Create a convolutional block with two convolutional layers,
    each followed by BatchNorm and ReLU activation.

    Parameters:
    - in_channels: int, number of input channels
    - out_channels: int, number of output channels
    - kernel_size: int, size of the convolutional kernel (default: 3)
    - padding: int, padding added to the input (default: 1)

    Returns:
    - nn.Sequential: A sequential container with the defined layers
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
def create_decoder_block(in_channels, middle_channels, out_channels, kernel_size=3, padding=1):
    """
    Create a decoder block with two convolutional layers,
    each followed by BatchNorm and ReLU activation.

    Parameters:
    - in_channels: int, number of input channels
    - middle_channels: int, number of channels after the first convolution
    - out_channels: int, number of output channels
    - kernel_size: int, size of the convolutional kernel (default: 3)
    - padding: int, padding added to the input (default: 1)

    Returns:
    - nn.Sequential: A sequential container with the defined layers
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(middle_channels),
        nn.ReLU(),
        nn.Conv2d(middle_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # # Encoder
        # vgg16 = models.vgg16(pretrained=True)
        # self.vgg16 = nn.Sequential(*list(vgg16.features.children()))
        self.attn = AttentionBlock(64)  # 添加注意力模块
        self.enc1 = create_conv_block(3, 16)
        self.enc2 = create_conv_block(16, 32)
        self.enc2plus = create_conv_block(32, 32)
        self.enc3 = create_conv_block(32, 64)

        self.enc3plus = create_conv_block(64, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = create_decoder_block(64, 32, 32)
        self.dec1plus = create_decoder_block(32, 32, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = create_decoder_block(32, 16, 16)
        self.dec2plus = create_decoder_block(16, 16, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x3 = self.enc2plus(x3)
        x3 = self.enc2plus(x3)
        x4 = self.pool(x3)
        x5 = self.enc3(x4)
        x5 = self.enc3plus(x5)
        x5 = self.attn(x5)

        # Decoder
        x6 = self.up1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x7 = self.dec1(x6)
        x7 = self.dec1plus(x7)

        x8 = self.up2(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x9 = self.dec2(x8)
        x9 = self.dec2plus(x9)
        x10 = torch.sigmoid(self.final(x9))
        return x10

class laneDetecteur():
    """
    CustomYOLOv8Model integrates the YOLOv8n (small) backbone into a custom model
    for a specific deep learning task (e.g., object detection).

    Attributes:
        yolo_model (YOLO): The YOLOv8n backbone model loaded with pretrained weights.

    Methods:
        forward(x): Defines the forward pass of the model using YOLOv8n.
        compute_loss(predictions, targets): Computes the loss between predictions and targets.
        predict(x): Makes predictions using the YOLOv8n backbone.
    """

    def __init__(self, pretrained=True, yolo_type="Segmentation"):
        """
        Initializes a new instance of CustomYOLOv8Model with the YOLOv8n backbone.

        Args:
            pretrained (bool): Whether to load pretrained weights for YOLOv8n.
        """
        super(laneDetecteur, self).__init__()
        self.working_dir = "./"
        # Load YOLOv8n model from Ultralytics (pretrained by default)
        self.yolo_type = yolo_type
        path = self.working_dir +'lane'+ yolo_type + '.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = torch.load(path, map_location=self.device)
        self.yolo_model = self.yolo_model.to(self.device)

    def display_lines(self, image, lines):
        lines_image = np.zeros_like(image)
        limit = 10000
        #make sure array isn't empty
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                # Check if coordinates are within the bounds of the image
                if -limit <= x1 < limit and -limit <= x2 < limit and -limit <= y1 < limit and -limit <= y2 < limit:
                    # Coordinates are within bounds, proceed with drawing the line
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                # else:
                #     # Coordinates are out of bounds
                #     print("Coordinates are out of range:", x1, y1, x2, y2)
                else: #defaut line, to control the variable
                    cv2.line(image, (1, 2), (1, 1), (255, 0, 0), 10)
        return lines_image

    def average(self, image, lines):
        left = []
        right = []
        if lines is None:  # Check if lines were detected
            return None
        if lines is not None:
            for line in lines:
                #print(line)
                x1, y1, x2, y2 = line.reshape(4)
                #fit line to points, return slope and y-int
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                #print(parameters)
                slope = parameters[0]
                y_int = parameters[1]
                #lines on the right have positive slope, and lines on the left have neg slope
                if slope < 0:
                    left.append((slope, y_int))
                else:
                    right.append((slope, y_int))

        #takes average among all the columns (column0: slope, column1: y_int)
        if left == []:
            left_avg = [1, 1]
        else:
            left_avg = np.average(left, axis=0)
        if right == []:
            right_avg = [1, 1]
        else:
            right_avg = np.average(right, axis=0)


        # print("right: ", right_avg, "left: ", left_avg)
        #create lines based on averages calculates

        left_line = self.make_points(image, left_avg)
        right_line = self.make_points(image, right_avg)
        return np.array([left_line, right_line])

    def make_points(self, image, average):
        # print("average: ", average)
        slope, y_int = average
        y1 = image.shape[0]
        #how long we want our lines to be --> 3/5 the size of the image
        y2 = int(y1 * (3/5))
        #determine algebraically
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
        return np.array([x1, y1, x2, y2])
    def region_of_detection(self, image, yolo_results):
            mask_size = image.shape
            mask = np.zeros(mask_size, dtype=np.uint8)
            # For each bounding box, draw a filled rectangle (1 for object area, 0 for background)
            for result in yolo_results:
                boxes = result.boxes.xyxy
                for box in boxes:
                    xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    # Ensure the box is within the image dimensions
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(mask_size[1], xmax)
                    ymax = min(mask_size[0], ymax)

                    # Draw a filled rectangle on the mask
                    mask[ymin:ymax, xmin:xmax] = 255  # Set the region inside the bounding box to 255
                    # cv2.imwrite(self.working_dir + "result_binary.jpeg", mask)
                    # print("mask: ", mask)
            return mask
    def region(self, original_image, mask):
        """
        使用掩码处理原始图像，返回处理后的图像
        :param original_image: 原始图像 (PIL 图像对象)
        :param mask: 二值掩码图像，0 和 255 (PIL 图像对象)
        :return: 处理后的图像
        """
        # 将原始图像和掩码转换为 numpy 数组
        original_array = np.array(original_image)
        mask_array = np.array(mask)

        # 确保掩码是二值图像，255为前景，0为背景
        mask_array = mask_array.astype(np.uint8)

        # 处理掩码：将掩码为0的部分变为黑色（或透明）
        result_array = original_array.copy()
        result_array[mask_array == 0] = 0  # 将背景部分设为黑色

        # 将处理后的数组转换回图像并返回
        result_image = Image.fromarray(result_array)
        return result_image

    def predict_angle_realtime(self, frame):
        """
        Makes real-time predictions using YOLOv8n on live video feed from Raspberry Pi Camera.
        :param camera: The picamera.PICamera object passed from outside.
        :param output: Whether to save the output video to a file (default is False).
        """
        if self.yolo_type == "Detection":
            print("This function is only for segmentation")
            return None
        # Set the resolution and framerate if not already set
        # camera.resolution = (640, 480)  # You can adjust this as needed
        # camera.framerate = 30  # Set framerate

        try:
            # Capture a frame from the camera
            # frame = camera.capture_array()
            # Process the frame
            copy = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            copy[:frame.shape[0] // 2, :] = 0
            isolated = self.get_isolated(copy, None)
            lines = cv2.HoughLinesP(isolated, 1, np.pi / 180, 2, np.array([]), minLineLength=40, maxLineGap=2)
            averaged_lines = self.average(copy, lines)

            # Return the intersection of the two lines
            current_x = frame.shape[1] / 2
            current_y = frame.shape[0]
            angle_degrees = 90
            limit = 10000
            if averaged_lines is not None and len(averaged_lines) == 2:
                x1, y1, x2, y2 = averaged_lines[0]
                x3, y3, x4, y4 = averaged_lines[1]
                if all(-limit < x < limit and -limit < y < limit for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]):
                    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    if denominator != 0:
                        # Calculate intersection point
                        intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
                        intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

                        # Calculate the angle in degrees
                        angle_radians = - math.atan2(intersection_y - current_y, intersection_x - current_x)
                        angle_degrees = math.degrees(angle_radians)

            return angle_degrees
        finally:
            pass

    def predict_single_image(self, frame):
        """
        处理直接传递的帧（frame）并进行预测，生成二值化图片。

        参数：
        - frame: 输入帧，可以是 numpy 数组或 PIL.Image。
        - model: 训练好的模型。
        - device: 设备类型（如 'cuda' 或 'cpu'）。

        返回：
        - output_image: 预测后的二值化图片（PIL.Image）。
        """
        # 检查输入帧类型并转换为 PIL.Image
        model = self.yolo_model
        device = self.device
        if isinstance(frame, np.ndarray):  # 如果是 numpy 数组
            image = Image.fromarray(frame).convert("RGB")
        elif isinstance(frame, Image.Image):  # 如果是 PIL.Image
            image = frame.convert("RGB")
        else:
            raise ValueError("输入帧的类型必须是 numpy 数组或 PIL.Image。")

        original_size = image.size

        # 转换图片
        input_image = F.resize(image, (256, 256))  # 调整大小
        input_tensor = F.to_tensor(input_image).unsqueeze(0).to(device)  # 转为张量并放入设备

        # 预测
        with torch.no_grad():
            output = model(input_tensor)  # 模型预测
            output = output.squeeze().cpu().numpy()  # 转为 numpy 数组

        # 二值化处理
        output = (output > 0.5).astype(np.uint8)  # 阈值化，转换为0和1

        # 还原到原始大小
        output_image = Image.fromarray((output * 255).astype(np.uint8))  # 转为图片格式
        output_image = output_image.resize(original_size, Image.BILINEAR)  # 调整回原始大小

        return output_image

    def get_isolated(self, frame, yolo_results=None):
        """Traite une seule image de la vidéo."""
        # Appliquez les mêmes étapes de traitement d'image que pour une image statique
        # print("shape of mask: ", mask)
        copy = np.copy(frame)
        if(yolo_results == None):
            if(self.yolo_type == "Segmentation"):
                yolo_results = self.predict_single_image(frame)               # yolo_results = self.yolo_model.predict(source=frame, save=False, save_txt=False, stream=False, show=False)
            else:
                yolo_results = self.yolo_model.predict(source=frame, save=False, save_txt=False, stream=False, show=False, conf=0.1)
        # Assuming 'copy' is a valid image (NumPy array)
        edges = cv2.Canny(copy, 100, 150)
        # inverted_edges = cv2.bitwise_not(edges)
        if(self.yolo_type == "Segmentation"):
            isolated = self.region(edges, yolo_results)
            # cv2.imwrite(self.working_dir + "result_binary.jpeg", isolated)
        else:
            isolated = self.region_of_detection(edges, yolo_results)
        isolated = np.uint8(isolated)
        return isolated

class panneauxDetecteur():
    def __init__(self):
        self.working_dir = "./"
        self.confidence = 0.02
        path = self.working_dir + "panneauxDetection.pt"
        self.yolo_model = YOLO(path)

    def predict_video(self, video_path):
        # Charger la vidéo
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for your desired output format
        out = cv2.VideoWriter('output_panneaux.mp4', fourcc, fps, (frame_width, frame_height))

        # Process and write frames to the output video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.yolo_model(frame, conf=0.05)  # 直接传入原图，YOLOv8会自动进行预处理
            image = self.process_image(results)

            # 假设模型的输出 predictions 是一个形状为 (grid_size, grid_size, num_anchors, 5 + num_classes) 的张量
            # predictions 可能来自于模型的 forward pass

            out.write(image)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        classes = []
        for result in results:
            classes.append(result.boxes.cls)
        return classes

    def predict_real_time(self, frame):
        """
        Real-time prediction using YOLOv8 for object detection.
        :param: frame
        """
        # 定义输出视频的编码格式和保存路径（如果 output=True）
        # 开始实时处理
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 使用 YOLOv8 模型进行推理
            results = self.yolo_model(image, conf=self.confidence)
            classes = []  # List to store detected classes

            # 处理每个检测结果，绘制边界框和标签
            for result in results:
                for cls in result.boxes.cls:
                    classes.append(result.names[int(cls)])

            # Yield detected classes for further processing
            return classes
        finally:
           pass
class obstacleDetecteur():
    def __init__(self):
        self.working_dir = "./"
        self.confidence = 0.2
        path = self.working_dir + "obstacleDetection.pt"
        self.yolo_model = YOLO(path)

    def predict_real_time(self, frame):
        # 定义输出视频的编码格式和保存路径（如果 output=True）
        # 开始实时处理
        try:
            # 使用 YOLOv8 模型进行推理
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = self.yolo_model(frame, conf=self.confidence)
            # Collect classes detected in this frame
            # 处理每个检测结果，绘制边界框和标签
            distances = []
            for result in results:
                for box in result.boxes.xywh:
                    width = box[2]
                    distance = self.calculate_distance(width)
                    distances.append(distance)
            return distances
        finally:
            pass
    def calculate_distance(self, width):
        # 假设车辆的实际宽度为 1.8 米，相机焦距为 800 像素
        KNOWN_WIDTH = 0.018  # 车辆的已知宽度（米）
        FOCAL_LENGTH = 800  # 焦距（像素）

        return KNOWN_WIDTH * FOCAL_LENGTH / width
# In[ ]:

class panneauxObstacleDetecteur():
    def __init__(self):
        self.working_dir = "./"
        self.confidence = 0.2
        path = self.working_dir + "panneauxObstacleDetection.pt"
        self.yolo_model = YOLO(path)

    def predict_real_time(self, frame):
        # 定义输出视频的编码格式和保存路径（如果 output=True）
        # 开始实时处理
        try:
            # 使用 YOLOv8 模型进行推理
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = self.yolo_model(frame, conf=self.confidence)
            classes = []  # List to store detected classes
            # Collect classes detected in this frame
            # 处理每个检测结果，绘制边界框和标签
            distances = []
            for result in results:
                for box, cls in zip(result.boxes.xywh, result.boxes.cls):
                    classes.append(result.names[int(cls)])
                    if result.names[int(cls)] == "Obstacle":
                        width = box[2]
                        distance = self.calculate_distance(width)
                        distances.append(distance)
            return classes, distances
        finally:
            pass
    def calculate_distance(self, width):
        # 假设车辆的实际宽度为 1.8 米，相机焦距为 800 像素
        KNOWN_WIDTH = 0.018  # 车辆的已知宽度（米）
        FOCAL_LENGTH = 800  # 焦距（像素）

        return KNOWN_WIDTH * FOCAL_LENGTH / width

# model = obstacleDetecteur()
# model.predict_video("/content/obstacle.mp4")
# def count_generator(gen):
    # # 创建一个新的生成器副本
    # gen1, gen2 = itertools.tee(gen)
    # nombre_panneaux = sum(1 for _ in gen1)
    # return nombre_panneaux, gen2

@dataclass
class Message:
    angle: float
    panneaux: list
    distance_ia: list

    def __post_init__(self):
        self.nombre_panneaux = len(list(self.panneaux))
        self.nombre_obstacles = len(list(self.distance_ia))
        #print(f"angle: {self.angle}, classes: {self.panneaux}, distance of obstacle: {self.distance_ia}")
    def encode(self):
        # 使用json序列化，可以处理更复杂的数据结构
        return json.dumps({
            'angle': self.angle,
            'panneaux': self.panneaux,
            'distances': self.distance_ia,
            'nombre_panneaux': self.nombre_panneaux,
            'nombre_obstacles': self.nombre_obstacles}).encode('utf-8')

    @classmethod
    def decode(cls, encoded_data):
        # 解码方法，用于反序列化
        data = json.loads(encoded_data.decode('utf-8'))
        return cls(
            angle=data['angle'],
            panneaux=data['panneaux'],
            distance_ia=data['distances']
        )# In[ ]:

def real_time():
    # Start capturing from the webcam
    import picamera2
    camera = picamera2.Picamera2()
    camera_config = camera.create_preview_configuration(main={"size": (640, 480)})
    camera.configure(camera_config)
    camera.start()
    panneaux = panneauxDetecteur()
    obstacle = obstacleDetecteur()
    lane = laneDetecteur()
    server_ip = "127.0.0.1"
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_port = 8080
    try:
        #i = 1
        while True:  # Continuous capture and processing
            # Use concurrent.futures to run detection models in parallel
            #print(f"le {i} fois")
            #i += 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

                frame = camera.capture_array()
                try:
                    # Submit parallel tasks for each detection model
                    lane_future = executor.submit(lane.predict_angle_realtime, frame)
                    #print("lane")
                    panneaux_future = executor.submit(panneaux.predict_real_time, frame)
                    #print("panneaux")
                    obstacle_future = executor.submit(obstacle.predict_real_time, frame)
                    #print("obstacle")

                    # Wait and get results
                    clss_panneaux = panneaux_future.result(timeout=8)
                    print("clss_panneaux:", clss_panneaux)
                except concurrent.futures.TimeoutError:
                    print("panneaux detection timed out")
                except Exception as e:
                    print(f"An error occurred: {e}")

                try:
                    angle = lane_future.result(timeout=8)
                    print("angle:", angle)
                except concurrent.futures.TimeoutError:
                    print("lane segmentation task timed out")
                except Exception as e:
                    print(f"An error occurred: {e}")

                try:
                    distance = obstacle_future.result(timeout=8)
                    print("distance:", distance)
                except concurrent.futures.TimeoutError:
                    print("obstacle detection task timed out")
                except Exception as e:
                    print(f"An error occurred: {e}")

                # Create and send message
                message = Message(angle, clss_panneaux, distance).encode()
                del angle
                del clss_panneaux
                del distance
                del lane_future
                del panneaux_future
                del obstacle_future

                print("message: ", message)

                try:
                    client_socket.sendto(message, (server_ip, server_port))
                    del message
                    print("Message sent")
                    #print(f"angle: {angle}, classes: {clss_panneaux}, distance of obstacle: {distance}")
                except Exception as e:
                    print(f"Error sending message: {e}")
            del executor

    except KeyboardInterrupt:
        print("Stopping real-time processing...")

    finally:
        camera.stop()
        client_socket.close()
#        yield angle, clss_panneaux, distance

def real_time_cv2():
    # Start capturing from the webcam
    camera = picamera2.Picamera2()
    camera_config = camera.create_preview_configuration(main={"size": (640, 480)})
    camera.configure(camera_config)
    camera.start()
    panneaux = panneauxDetecteur()
    obstacle = obstacleDetecteur()
    lane = laneDetecteur()
    server_ip = "127.0.0.1"
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_port = 8080
    try:
        #i = 1
        while True:  # Continuous capture and processing
            # Use concurrent.futures to run detection models in parallel
            #print(f"le {i} fois")
            #i += 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

                frame = camera.capture_array()
                try:
                    # Submit parallel tasks for each detection model
                    lane_future = executor.submit(lane.predict_angle_realtime, frame)
                    #print("lane")
                    panneaux_future = executor.submit(panneaux.predict_real_time, frame)
                    #print("panneaux")
                    obstacle_future = executor.submit(obstacle.predict_real_time, frame)
                    #print("obstacle")

                    # Wait and get results
                    clss_panneaux = panneaux_future.result(timeout=8)
                    print("clss_panneaux:", clss_panneaux)
                except concurrent.futures.TimeoutError:
                    print("panneaux detection timed out")
                except Exception as e:
                    print(f"An error occurred: {e}")

                try:
                    angle = lane_future.result(timeout=8)
                    print("angle:", angle)
                except concurrent.futures.TimeoutError:
                    print("lane segmentation task timed out")
                except Exception as e:
                    print(f"An error occurred: {e}")

                try:
                    distance = obstacle_future.result(timeout=8)
                    print("distance:", distance)
                except concurrent.futures.TimeoutError:
                    print("obstacle detection task timed out")
                except Exception as e:
                    print(f"An error occurred: {e}")

                # Create and send message
                message = Message(angle, clss_panneaux, distance).encode()
                del angle
                del clss_panneaux
                del distance
                del lane_future
                del panneaux_future
                del obstacle_future

                print("message: ", message)

                try:
                    client_socket.sendto(message, (server_ip, server_port))
                    del message
                    print("Message sent")
                    #print(f"angle: {angle}, classes: {clss_panneaux}, distance of obstacle: {distance}")
                except Exception as e:
                    print(f"Error sending message: {e}")
            del executor

    except KeyboardInterrupt:
        print("Stopping real-time processing...")

    finally:
        camera.stop()
        client_socket.close()


def real_time_switch(camera_type='opencv'):
    if camera_type == 'picamera2':
        import picamera2
        camera = picamera2.Picamera2()
        camera_config = camera.create_preview_configuration(main={"size": (640, 480)})
        camera.configure(camera_config)
        camera.start()
    elif camera_type == 'opencv':
        camera = cv2.VideoCapture(0)  # 0 表示默认摄像头
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        raise ValueError("Invalid camera type. Use 'picamera2' or 'opencv'.")

    # panneaux = panneauxDetecteur()
    # obstacle = obstacleDetecteur()
    lane = laneDetecteur()
    panneaux_obstacle = panneauxObstacleDetecteur()
    server_ip = "127.0.0.1"
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_port = 8080

    try:
        while True:  # Continuous capture and processing
            if camera_type == 'picamera2':
                frame = camera.capture_array()
            elif camera_type == 'opencv':
                ret, frame = camera.read()
                if not ret:
                    print("Failed to capture frame")
                    break
            else:
                return

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                try:
                    # Submit parallel tasks for each detection model
                    lane_future = executor.submit(lane.predict_angle_realtime, frame)
                    # panneaux_future = executor.submit(panneaux.predict_real_time, frame)
                    # obstacle_future = executor.submit(obstacle.predict_real_time, frame)
                    panneaux_obstacle_future = executor.submit(panneaux_obstacle.predict_real_time, frame)
                    # Wait and get results
                    # clss_panneaux = panneaux_future.result(timeout=8)
                    clss_panneaux, distance = panneaux_obstacle_future.result(timeout=8)
                    # print("clss_distance:", clss_distance)
                    # print("clss_panneaux:", clss_panneaux)
                except concurrent.futures.TimeoutError:
                    print("panneaux detection timed out")
                    return
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return

                try:
                    angle = lane_future.result(timeout=8)
                    print("angle:", angle)
                except concurrent.futures.TimeoutError:
                    print("lane segmentation task timed out")
                except Exception as e:
                    print(f"An error occurred: {e}")

                # try:
                    # distance = obstacle_future.result(timeout=8)
                    # print("distance:", distance)
                # except concurrent.futures.TimeoutError:
                    # print("obstacle detection task timed out")
                # except Exception as e:
                    # print(f"An error occurred: {e}")

                # Create and send message
                message = Message(angle, clss_panneaux, distance).encode()
                print("message: ", message)

                try:
                    client_socket.sendto(message, (server_ip, server_port))
                    print("Message sent")
                except Exception as e:
                    print(f"Error sending message: {e}")

    except KeyboardInterrupt:
        print("Stopping real-time processing...")

    finally:
        if camera_type == 'picamera2':
            camera.stop()
        elif camera_type == 'opencv':
            camera.release()
        client_socket.close()
# if __name__ == "__main__":
real_time_switch("picamera2")
