import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_json_files(folder_path):
    """加载文件夹下所有 JSON 文件."""
    json_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.json')]
    json_files.sort()  # 按文件名排序
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            data.append(json.load(f))
    return data

def extract_keypoints(data):
    """提取人体、脸部、和手部关键点."""
    all_keypoints = []
    for frame_data in data:
        people = frame_data.get('people', [])
        if people:
            person = people[0]
            pose_keypoints = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
            face_keypoints = np.array(person.get('face_keypoints_2d', [])).reshape(-1, 3)
            hand_left_keypoints = np.array(person.get('hand_left_keypoints_2d', [])).reshape(-1, 3)
            hand_right_keypoints = np.array(person.get('hand_right_keypoints_2d', [])).reshape(-1, 3)
            all_keypoints.append((pose_keypoints, face_keypoints, hand_left_keypoints, hand_right_keypoints))
        else:
            all_keypoints.append((np.zeros((25, 3)), np.zeros((70, 3)), np.zeros((21, 3)), np.zeros((21, 3))))  # 填充零
    return all_keypoints

def plot_keypoints(pose_keypoints, face_keypoints, hand_left_keypoints, hand_right_keypoints):
    """绘制单帧人体、脸部、和手部关键点."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 绘制骨骼（COCO 模型的连接）
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), 
        (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
        (0, 15), (15, 17), (0, 16), (16, 18), (11, 24), (11, 22), 
        (14, 21), (14, 19)
    ]
    for start, end in skeleton:
        if pose_keypoints[start, 2] > 0 and pose_keypoints[end, 2] > 0:  # 仅连接置信度高的点
            ax.plot(
                [pose_keypoints[start, 0], pose_keypoints[end, 0]],
                [-pose_keypoints[start, 1], -pose_keypoints[end, 1]],
                c='#7898e1'  # 蓝色骨骼
            )

    # 绘制关键点
    for keypoints, color in [
        (pose_keypoints, '#f89588'),  # 红色
        (face_keypoints, '#9192ab'),  # 绿色
        (hand_left_keypoints, '#f8cb7f'),  # 紫色
        (hand_right_keypoints, '#7cd6cf')  # 橙色
    ]:
        for x, y, c in keypoints:
            if c > 0:  # 绘制置信度高的点
                ax.scatter(x, -y, s=10, c=color)

    # 设置绘图区域
    ax.set_xlim(0, 640)  # 假设图片分辨率为 640x480
    ax.set_ylim(-480, 0)
    ax.axis('off')

    # 保存帧为图片
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(image)

def save_frames_as_pdf(folder_path, output_pdf):
    """将所有帧保存为图片，并拼接成一张长图，导出 PDF."""
    data = load_json_files(folder_path)
    keypoints = extract_keypoints(data)

    frame_images = [plot_keypoints(*keypoints[i]) for i in range(len(keypoints))]

    # 计算拼接后图像的总宽度和最大高度
    total_width = sum(img.width for img in frame_images)
    max_height = max(img.height for img in frame_images)

    # 创建空白画布
    new_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    # 依次拼接图片
    x_offset = 0
    for img in frame_images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # 保存为 PDF
    new_image.save(output_pdf, "PDF", resolution=100.0)
    print(f"PDF 已生成: {output_pdf}")

# 使用方法
folder_path = rf'C:\Users\18801\Desktop\LZQ\IJCAI\data\happy\4-6'  # 替换为 JSON 文件的文件夹路径
output_pdf_path = folder_path + rf'\output_frames.pdf'  # 输出 PDF 文件路径
save_frames_as_pdf(folder_path, output_pdf_path)
