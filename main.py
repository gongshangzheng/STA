#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math

import cv2
#import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import picamera2
# import onnx
# import onnxruntime as ort
from ultralytics import YOLO

# In[ ]:


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
        path = self.working_dir +'lane'+ yolo_type + '.pt'
        self.yolo_model = YOLO(path)

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

    def region(self, image, yolo_results):
        """
        Convert the YOLO segmentation result to a bresults = model.predict(source=img.copy(), save=True, save_txt=False, stream=True)nary mask with only 0 and 1 values.

        Args:
            image_path (str): Path to the image for inference.
            save_dir (str): Directory where the binary mask image will be saved.
        """
        # Perform inference using YOLOv8 (segmentation)
        masked = np.zeros_like(image.shape)
        for result in yolo_results:
            # get array results
            if result.masks is None:
                break
            masks = result.masks.data
            boxes = result.boxes.data
            # extract classes
            clss = boxes[:, 5]
            # get indices of results where class is 0 (lane)
            lane_indices = torch.where(clss == 0)
            # use these indices to extract the relevant masks
            lane_masks = masks[lane_indices]
            if lane_masks.size(0) > 0:
                lane_mask = torch.any(lane_masks, dim=0).int() * 255
                lane_mask = lane_mask.cpu().numpy().astype(np.uint8)  # Convert to numpy for OpenCV

                # Resize lane_mask to match the original image if necessary
                lane_mask_resized = cv2.resize(lane_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Apply mask to original image (ensuring lane_mask_resized has 3 channels if needed)
                # if len(lane_mask_resized.shape) == 2:  # Make sure it's 3-channel to match the image
                #     lane_mask_resized = cv2.cvtColor(lane_mask_resized, cv2.COLOR_GRAY2BGR)
                masked = cv2.bitwise_and(image, lane_mask_resized)

                # Save the resulting masked image
                # cv2.imwrite(self.working_dir + "result_binary.jpeg", mask)
            # else:
            #     print("No lane masks found for the class specified.")

        return masked

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

    def predict_angle(self, video_path = "/content/Lane.mp4", output=False):
        """
        Makes predictions using the YOLOv8n
        """
        if(self.yolo_type == "Detection"):
            print("This function is only for segmentation")
            return None
        # Charger la vidéo
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if output:
            #output the video
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for your desired output format
            out = cv2.VideoWriter('output_direction.mp4', fourcc, fps, (frame_width, frame_height))

        # get two lines
        lines = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            copy = np.copy(frame)
            copy[:frame_height // 2, :] = 0
            isolated = self.get_isolated(copy, None)
            lines = cv2.HoughLinesP(isolated, 1, np.pi/180, 2, np.array([]), minLineLength=40, maxLineGap=2)
            averaged_lines = self.average(copy, lines)
            #return the intersection of the two lines
            current_x = frame.shape[1] / 2
            current_y = frame.shape[0]
            angle = 90
            limit = 10000
            if averaged_lines is not None and len(averaged_lines) == 2:

                x1, y1, x2, y2 = averaged_lines[0]
                x3, y3, x4, y4 = averaged_lines[1]
                if all(-limit < x < limit and -limit < y < limit for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]):
                    # Your code here
                    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    if denominator != 0:
                        # Calculate intersection point
                        intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
                        intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

                        # Calculate the angle in degrees
                        angle_radians = - math.atan2(intersection_y - current_y, intersection_x - current_x)
                        angle_degrees = math.degrees(angle_radians)

                        # Display the guiding line on the frame
                        if output:
                            longueur = 3000
                            end_x = current_x + longueur * math.cos(angle_radians)
                            end_y = current_y - longueur * math.sin(angle_radians)
                            cv2.line(frame, (int(current_x), int(current_y)), (int(end_x), int(end_y)), (0, 255, 0), 2)
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            cv2.line(frame, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 0), 2)
                            cv2.putText(frame, f"Angle: {angle_degrees:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            out.write(frame)

                    # Display angle on frame

            # print("angle: ", angle_degrees)
            # print("radian: ", angle_radians)



        cap.release()
        if output:
            out.release()
            cv2.destroyAllWindows()
        return angle_degrees

    def predict_angle_realtime(self, camera, output=False):
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

        if output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = camera.resolution[0]
            frame_height = camera.resolution[1]
            fps = camera.framerate
            out = cv2.VideoWriter('output_direction_realtime.mp4', fourcc, fps, (frame_width, frame_height))
        try:
            while True:
                # Capture a frame from the camera
                frame = camera.capture_array() 
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

                            # Display the guiding line on the frame
                            if output:
                                longueur = 3000
                                end_x = current_x + longueur * math.cos(angle_radians)
                                end_y = current_y - longueur * math.sin(angle_radians)
                                cv2.line(frame, (int(current_x), int(current_y)), (int(end_x), int(end_y)), (0, 255, 0), 2)
                                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                                cv2.line(frame, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 0), 2)
                                cv2.putText(frame, f"angle: {angle_degrees:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                out.write(frame)

                # Show the frame in real-time
                if output:
                    cv2.imshow("real-time lane angle prediction", frame)

                # Yield the angle prediction
                yield angle_degrees

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:

            camera.stop()
            # Release resources
            if output:
                out.release()
            cv2.destroyAllWindows()


    def get_isolated(self, frame, yolo_results=None):
        """Traite une seule image de la vidéo."""
        # Appliquez les mêmes étapes de traitement d'image que pour une image statique
        # print("shape of mask: ", mask)
        copy = np.copy(frame)
        if(yolo_results == None):
            if(self.yolo_type == "Segmentation"):
                yolo_results = self.yolo_model.predict(source=frame, save=False, save_txt=False, stream=False, show=False)
                # yolo_results = self.yolo_model.predict(source=frame, save=False, save_txt=False, stream=False, show=False)
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

    def process_image(self, results):
        # 获取预测的边界框、类别和置信度
        for result in results:
            predicted_boxes = result.boxes.xyxy  # 获取预测的边界框坐标 (x1, y1, x2, y2)
            predicted_confidences = result.boxes.conf  # 置信度
            predicted_classes = result.boxes.cls  # 类别索引
            # print(results)
            image = result.orig_img

            # 获取类别名称
            class_names = result.names  # 获取类别名称字典

            # 将边界框和标签绘制回原图
            for box, conf, cls in zip(predicted_boxes, predicted_confidences, predicted_classes):
                x1, y1, x2, y2 = box  # 提取边界框的坐标

                # 将边界框坐标转换为图像坐标
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制标签
                label = f'{class_names[int(cls)]}: {conf:.2f}'  # 获取类别名称并显示置信度
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def predict_real_time(self, camera, config, output=True):
        """
        Real-time prediction using YOLOv8 for object detection.

        :param camera: PiCamera object (from picamera library) for real-time video feed
        :param output: Whether to save the video to file (default True)
        """
        # 定义输出视频的编码格式和保存路径（如果 output=True）
        if output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output_real_time.mp4', fourcc, 20, config["main"]["size"])


        # 开始实时处理
        try:
            while True:
                frame = camera.capture_array()
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 使用 YOLOv8 模型进行推理
                results = self.yolo_model(image, conf=self.confidence)
                classes = []  # List to store detected classes

                # 处理每个检测结果，绘制边界框和标签
                for result in results:
                    if output:
                        boxes = result.boxes.xyxy  # 获取边界框坐标 (x1, y1, x2, y2)
                        confidences = result.boxes.conf  # 获取置信度

                        for box, conf, cls in zip(boxes, confidences, result.boxes.cls):
                            x1, y1, x2, y2 = map(int, box)
                            label = f'{result.names[int(cls.item())]}: {conf.item():.2f}'
                            # 绘制边界框和标签
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            classes.append(result.names[int(cls)])

                # 显示帧率并显示帧图像
                if output:
                    cv2.imshow("Real-Time YOLOv8", image)
                    out.write(image)

                # 按下 'q' 键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Yield detected classes for further processing
                yield classes
        finally:
            camera.stop()
            # 释放资源
            if output:
                out.release()
            cv2.destroyAllWindows()
class obstacleDetecteur():
    def __init__(self):
        self.working_dir = "./"
        self.confidence = 0.2
        path = self.working_dir + "obstacleDetection.pt"
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
        out = cv2.VideoWriter('output_obstacle.mp4', fourcc, fps, (frame_width, frame_height))

        # Process and write frames to the output video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.yolo_model(frame, conf=self.confidence)  # 直接传入原图，YOLOv8会自动进行预处理
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

    def process_image(self, results):
        # 获取预测的边界框、类别和置信度
        for result in results:
            predicted_boxes = result.boxes.xyxy  # 获取预测的边界框坐标 (x1, y1, x2, y2)
            predicted_confidences = result.boxes.conf  # 置信度
            predicted_classes = result.boxes.cls  # 类别索引
            # print(results)
            image = result.orig_img

            # 获取类别名称
            class_names = result.names  # 获取类别名称字典

            # 将边界框和标签绘制回原图
            for box, conf, cls in zip(predicted_boxes, predicted_confidences, predicted_classes):
                x1, y1, x2, y2 = box  # 提取边界框的坐标

                # 将边界框坐标转换为图像坐标
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制标签
                label = f'{class_names[int(cls)]}: {conf:.2f}'  # 获取类别名称并显示置信度
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def predict_real_time(self, camera, config, output=True):
        # 定义输出视频的编码格式和保存路径（如果 output=True）
        if output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output_real_time.mp4', fourcc, 20, config["main"]["size"])

        # 开始实时处理
        try:
            while True:
                frame = camera.capture_array()
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
                    if output:
                        boxes = result.boxes.xyxy  # 提取边界框 (x1, y1, x2, y2)
                        confidences = result.boxes.conf  # 置信度

                        for box, conf, cls in zip(boxes, confidences, result.boxes.cls):
                                x1, y1, x2, y2 = map(int, box)
                                label = f'{result.names[int(cls.item())]}: {conf.item():.2f}'

                                # 绘制边界框和标签
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 显示帧率
                if output:
                    cv2.imshow("Real-Time YOLOv8", frame)
                    out.write(frame)

                    # 按下 'q' 键退出循环
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                yield distances

        finally:

            camera.stop()
            # 释放资源
            if output:
                out.release()
            cv2.destroyAllWindows()

    def calculate_distance(self, width):
        # 假设车辆的实际宽度为 1.8 米，相机焦距为 800 像素
        KNOWN_WIDTH = 1.8  # 车辆的已知宽度（米）
        FOCAL_LENGTH = 800  # 焦距（像素）

        return KNOWN_WIDTH * FOCAL_LENGTH / width

# model = obstacleDetecteur()
# model.predict_video("/content/obstacle.mp4")


# In[ ]:

def real_time():
    # Start capturing from the webcam
    camera = picamera2.Picamera2()
    camera_config = camera.create_preview_configuration(main={"size": (640, 480)})
    camera.configure(camera_config)
    camera.start()
    panneaux = panneauxDetecteur()
    obstacle = obstacleDetecteur()
    lane = laneDetecteur()
    for clss_panneaux, angle,distance in zip(panneaux.predict_real_time(camera, camera_config, output=False), lane.predict_angle_realtime(camera, output=False), obstacle.predict_real_time(camera, camera_config, output=False)):
        print("angle: ", angle,"classes: ", clss_panneaux, "distance of obstacle: ", distance)
#        yield angle, clss_panneaux, distance


if __name__ == "__main__":
    real_time()

