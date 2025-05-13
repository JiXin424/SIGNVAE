import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

class PoseVisualizer:
    def __init__(self):
        # 身体骨架连接
        self.body_pairs = [
            [1, 0], [1, 2], [2, 3], [3, 4],
            [1, 5], [5, 6], [6, 7],
            [1, 8], [8, 9], [9, 10], [10, 11],
            [8, 12], [12, 13], [13, 14],
            [0, 15], [15, 17], [0, 16], [16, 18]
        ]
        
        # 面部骨架连接
        self.face_pairs = [
            # 轮廓
            [0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10],
            [10,11], [11,12], [12,13], [13,14], [14,15], [15,16],
            # 眉毛
            [17,18], [18,19], [19,20], [20,21],
            [22,23], [23,24], [24,25], [25,26],
            # 鼻子
            [27,28], [28,29], [29,30],
            [31,32], [32,33], [33,34], [34,35],
            # 眼睛
            [36,37], [37,38], [38,39], [39,40], [40,41], [41,36],
            [42,43], [43,44], [44,45], [45,46], [46,47], [47,42],
            # 嘴巴
            [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], [55,56], [56,57], [57,58], [58,59], [59,48]
        ]
        
        # 手部骨架连接
        self.hand_pairs = [
            [0,1], [1,2], [2,3], [3,4],           # 拇指
            [0,5], [5,6], [6,7], [7,8],           # 食指
            [0,9], [9,10], [10,11], [11,12],      # 中指
            [0,13], [13,14], [14,15], [15,16],    # 无名指
            [0,17], [17,18], [18,19], [19,20]     # 小指
        ]
        
        # 初始化matplotlib
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title('OpenPose Skeleton Visualization')
        
        # 读取所有JSON文件
        self.json_files = sorted(glob.glob("*.json"))
        self.current_frame = 0
        
    def load_keypoints(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if data['people'] and len(data['people']) > 0:
            person = data['people'][0]
            return {
                'pose': np.array(person['pose_keypoints_2d']),
                'face': np.array(person['face_keypoints_2d']) if 'face_keypoints_2d' in person else None,
                'hand_left': np.array(person['hand_left_keypoints_2d']) if 'hand_left_keypoints_2d' in person else None,
                'hand_right': np.array(person['hand_right_keypoints_2d']) if 'hand_right_keypoints_2d' in person else None
            }
        return None

    def draw_skeleton(self, keypoints, pairs, color='g', confidence_threshold=0):
        if keypoints is None:
            return
            
        # 绘制骨架连接
        for pair in pairs:
            p1_idx = pair[0] * 3
            p2_idx = pair[1] * 3
            
            if (keypoints[p1_idx + 2] > confidence_threshold and 
                keypoints[p2_idx + 2] > confidence_threshold):
                self.ax.plot([keypoints[p1_idx], keypoints[p2_idx]],
                           [keypoints[p1_idx + 1], keypoints[p2_idx + 1]],
                           f'{color}-', linewidth=1)
        
        # 绘制关键点
        for i in range(0, len(keypoints), 3):
            if keypoints[i + 2] > confidence_threshold:
                self.ax.plot(keypoints[i], keypoints[i + 1], f'{color}o', markersize=2)

    def update(self, frame):
        self.ax.clear()
        self.ax.set_xlim(0, 1920)
        self.ax.set_ylim(1080, 0)
        
        if frame < len(self.json_files):
            data = self.load_keypoints(self.json_files[frame])
            
            if data is not None:
                # 绘制身体骨架
                self.draw_skeleton(data['pose'], self.body_pairs, 'g')
                # 绘制面部骨架
                self.draw_skeleton(data['face'], self.face_pairs, 'b')
                # 绘制左手骨架
                self.draw_skeleton(data['hand_left'], self.hand_pairs, 'r')
                # 绘制右手骨架
                self.draw_skeleton(data['hand_right'], self.hand_pairs, 'y')
                
        self.ax.set_title(f'Frame: {frame + 1}/{len(self.json_files)}')
        
    def animate(self):
        """创建动画"""
        if not self.json_files:
            print("当前目录下没有找到JSON文件！")
            return
        
        print(f"找到 {len(self.json_files)} 个JSON文件")
        anim = FuncAnimation(self.fig, self.update,
                           frames=len(self.json_files),
                           interval=100,  # 每帧之间的间隔时间（毫秒）
                           repeat=False)
        plt.show()

    def show_single_frame(self, frame_idx):
        """显示单个帧"""
        if frame_idx < len(self.json_files):
            self.update(frame_idx)
            plt.show()
        else:
            print(f"帧索引超出范围！总帧数：{len(self.json_files)}")

def main():
    visualizer = PoseVisualizer()
    
    # 选择显示模式
    print("请选择显示模式：")
    print("1. 动画模式（连续播放所有帧）")
    print("2. 单帧模式（显示指定帧）")
    
    choice = input("请输入选择（1或2）：")
    
    if choice == '1':
        visualizer.animate()
    elif choice == '2':
        frame_idx = int(input(f"请输入要显示的帧序号（0-{len(visualizer.json_files)-1}）："))
        visualizer.show_single_frame(frame_idx)
    else:
        print("无效的选择！")

if __name__ == "__main__":
    main()