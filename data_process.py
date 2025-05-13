import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# =============== 1. 解析 JSON 文件，提取关键点 ===============
def parse_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not data["people"]:
        return None  # 如果没有检测到人体，返回 None
    
    person = data["people"][0]  # 取第一个检测到的人
    
    pose = np.array(person["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    hand_left = np.array(person["hand_left_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    hand_right = np.array(person["hand_right_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    face = np.array(person["face_keypoints_2d"], dtype=np.float32).reshape(-1, 3)

    # watch = np.array(pose + hand_left + hand_right + face).reshape(-1, 3)
    # print(watch)
    keypoints = np.vstack([pose, hand_left, hand_right, face])
    keypoints = keypoints[:, :2]

    return keypoints  # 3D: x, y, confidence


# =============== 2. 解析一个文件夹中的动作序列 ===============
def load_motion_sequence(folder_path, max_frames=150):
    json_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')])
    
    frames = []
    for file in json_files:
        keypoints = parse_json(file)
        if keypoints is not None:
            frames.append(keypoints)
    
    frames = np.array(frames)  # (num_frames, num_points, 2)

    # 归一化关键点坐标
    frames = normalize_motion_sequence(frames)

    # 统一帧数
    if frames.shape[0] < max_frames:
        # 插值补全
        frames = np.pad(frames, ((0, max_frames - frames.shape[0]), (0, 0), (0, 0)), mode='constant')
    else:
        # 截断
        frames = frames[:max_frames]
    
    return frames  # (max_frames, num_points, 2)


# =============== 3. 归一化关键点，使相对位置保持不变 ===============
def normalize_motion_sequence(frames, width= 320, height= 320):
    if frames.size == 0:
        return frames  # 直接返回空数据
    
    normalized_video_keypoints = frames / np.array([width, height])
    return normalized_video_keypoints
        

def denormalize_motion_sequence(normalized_frames, width= 320, height= 320):
    if normalized_frames.size == 0:
        return normalized_frames  # 直接返回空数据
    
    denormalized_video_keypoints = normalized_frames * np.array([width, height])
    return denormalized_video_keypoints

# =============== 4. 读取整个数据集 ===============
def load_dataset(data_path, max_frames=150):
    motions = []
    labels = []
    
    with open(os.path.join(data_path, "file_mapping.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        folder_id, word = parts[0], parts[1]
        folder_path = os.path.join(data_path, folder_id)
        
        if os.path.exists(folder_path):
            motion_data = load_motion_sequence(folder_path, max_frames)
            motions.append(motion_data)
            labels.append(word)
    
    return np.array(motions), labels  # (num_samples, max_frames, num_points, 2)




# =============== 5. 显示 OpenPose 骨架动画 ===============
def visualize_motion_sequence(sequence, output_path="./output_demo"):
    """
    以视频的方式展示 OpenPose 估计的骨架序列，并保存帧到 output_path
    :param sequence: (num_frames, num_points, 2) 关键点数据
    :param output_path: 保存帧的文件夹
    """
    os.makedirs(output_path, exist_ok=True)  # 创建输出目录

    # 拆分 sequence 数组
    pose_keypoints_list = sequence[:, :25, :]
    hand_left_keypoints_list = sequence[:, 25:46, :]
    hand_right_keypoints_list = sequence[:, 46:67, :]
    face_keypoints_list = sequence[:, 67:, :]

    for index, (pose_keypoints, hand_left_keypoints, hand_right_keypoints, face_keypoints) in enumerate(zip(pose_keypoints_list, hand_left_keypoints_list, hand_right_keypoints_list, face_keypoints_list)):
        fig, ax = plt.subplots(figsize=(6, 4))
    
        # 绘制骨骼（COCO 模型的连接）
        skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), 
            (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
            (0, 15), (15, 17), (0, 16), (16, 18), (11, 24), (11, 22), 
            (14, 21), (14, 19)
        ]
        for start, end in skeleton:
            a = np.all(pose_keypoints[start] < 20)
            b = np.all(pose_keypoints[end] < 20)
            if a or b:
                continue  # 如果是，则跳过此连接
            ax.plot(
                [pose_keypoints[start, 0], pose_keypoints[end, 0]],
                [-pose_keypoints[start, 1], -pose_keypoints[end, 1]],  # 注意这里将y坐标取反，可能是为了匹配图像坐标系统
                c='#7898e1'  # 蓝色骨骼
            )
            # 绘制关键点
        for keypoints, color in [
            (pose_keypoints, '#f89588'),  # 红色
            (face_keypoints, '#9192ab'),  # 绿色
            (hand_left_keypoints, '#f8cb7f'),  # 紫色
            (hand_right_keypoints, '#7cd6cf')  # 橙色
        ]:
            for x, y in keypoints:
                ax.scatter(x, -y, s=10, c=color)

        # 设置绘图区域
        ax.set_xlim(0, 320)  # 假设图片分辨率为 640x480
        ax.set_ylim(-240, 0)
        ax.axis('off')

        # 保存帧为图片
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        image = Image.fromarray(image)
        image.save(f"{output_path}/demo_{index:03d}.png")
    return 0


# =============== 6. 运行 Demo ===============
if __name__ == "__main__":
    data_path = "data"  # 你的数据目录
    motions, labels = load_dataset(data_path)

    print(f"数据加载完成，共 {len(motions)} 个样本")

    # 以第一组动作序列为例，生成视频
    if len(motions) > 0:
        raw_motion = denormalize_motion_sequence(motions[0])
        visualize_motion_sequence(raw_motion)