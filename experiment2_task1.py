import cv2
import numpy as np
import pyautogui
import json
import time
import tkinter as tk
from gaze_tracking import GazeTracking
from filterpy.kalman import KalmanFilter
import os
import random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# 导入基础眼动追踪模块
from optimized_click import (
    get_gaze_coordinates,
    create_kalman_filter_acc,
    load_calibration,
    low_pass_filter,
    distance,
    moving_average,
    is_in_dead_zone
)

# ==============================
# 实验配置
SUBJECT_NAME = "gff"  # 修改为受试者姓名
USE_IMPLICIT_CALIBRATION = True  # 是否使用隐式校准
EXPERIMENT_BUTTONS_FILE = "experiment_buttons.json"  # 实验按钮配置文件
CALIBRATION_FILE = "calibration-5.json"  # 校准文件

# 实验参数
DURATION_THRESHOLD = 1.0  # 停留阈值(秒)
DISTANCE_THRESHOLD = 100  # 稳定范围(像素)
COOLDOWN_AFTER_CLICK = 1.0  # 点击后冷却时长(秒)
MOVING_AVERAGE_WINDOW = 5  # 移动平均窗口大小
DEAD_ZONE = 5  # 死区范围（像素）
MIN_VELOCITY_THRESHOLD = 0.5  # 最小速度阈值
Y_OFFSET = -10  # Y轴偏移量

# 获取屏幕分辨率
screen_width, screen_height = pyautogui.size()

class ExperimentData:
    def __init__(self, subject_name, use_implicit_calibration):
        self.subject_name = subject_name
        self.use_implicit_calibration = use_implicit_calibration
        self.experiment_dir = f"E2T1_{subject_name}_{'implicit' if use_implicit_calibration else 'explicit'}"
        self.raw_data = []
        self.processed_data = []
        self.current_round = 1  # 从1开始，每个按钮一个round
        self.current_button = None
        self.start_time = None
        
        # 创建实验目录
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            
    def add_raw_data(self, gaze_x, gaze_y, button):
        """添加原始数据"""
        self.raw_data.append({
            'round': self.current_round,
            'button_id': button['id'],
            'button_name': button['name'],
            'button_x': button['x'] + button['width']//2,
            'button_y': button['y'] + button['height']//2,
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'timestamp': time.time() - self.start_time
        })
        
    def add_processed_data(self, click_x, click_y, button, hit):
        """添加处理后的数据"""
        self.processed_data.append({
            'round': self.current_round,
            'button_id': button['id'],
            'button_name': button['name'],
            'button_x': button['x'] + button['width']//2,
            'button_y': button['y'] + button['height']//2,
            'click_x': click_x,
            'click_y': click_y,
            'hit': hit,
            'timestamp': time.time() - self.start_time
        })
        
    def next_round(self):
        """进入下一轮"""
        self.current_round += 1
        
    def save_data(self):
        """保存实验数据"""
        # 保存原始数据
        raw_df = pd.DataFrame(self.raw_data)
        raw_df.to_csv(f"{self.experiment_dir}/raw_data.csv", index=False)
        
        # 保存处理后数据
        processed_df = pd.DataFrame(self.processed_data)
        processed_df.to_csv(f"{self.experiment_dir}/processed_data.csv", index=False)
        
        # 生成分析报告
        self.generate_report()
        
    def generate_report(self):
        """生成实验报告"""
        processed_df = pd.DataFrame(self.processed_data)
        raw_df = pd.DataFrame(self.raw_data)
        
        # 计算每轮的指标
        mae_by_round = []
        gaze_hit_rate_by_round = []
        
        # 定义采样间隔（每隔多少帧采样一次）
        SAMPLE_INTERVAL = 5
        # 定义按钮周围的范围（像素）
        BUTTON_MARGIN = 100
        
        for round_num in range(1, 13):  # 12个按钮，12个round
            round_data = processed_df[processed_df['round'] == round_num]
            round_raw_data = raw_df[raw_df['round'] == round_num]
            
            if len(round_data) > 0:  # 确保该轮有数据
                # 计算MAE
                mae = np.mean(np.sqrt(
                    (round_data['click_x'] - round_data['button_x'])**2 +
                    (round_data['click_y'] - round_data['button_y'])**2
                ))
                mae_by_round.append(mae)
                
                # 计算注视点命中率
                if len(round_raw_data) > 0:
                    # 获取当前按钮的位置
                    button_x = round_raw_data['button_x'].iloc[0]
                    button_y = round_raw_data['button_y'].iloc[0]
                    button_width = 100  # 按钮宽度
                    button_height = 100  # 按钮高度
                    
                    # 每隔SAMPLE_INTERVAL帧采样一次
                    sampled_data = round_raw_data.iloc[::SAMPLE_INTERVAL]
                    
                    # 计算在按钮周围范围内的点
                    in_range_points = 0
                    in_button_points = 0
                    
                    for _, row in sampled_data.iterrows():
                        gaze_x, gaze_y = row['gaze_x'], row['gaze_y']
                        
                        # 检查是否在按钮周围范围内
                        if (abs(gaze_x - button_x) <= button_width/2 + BUTTON_MARGIN and 
                            abs(gaze_y - button_y) <= button_height/2 + BUTTON_MARGIN):
                            in_range_points += 1
                            
                            # 检查是否在按钮区域内
                            if (abs(gaze_x - button_x) <= button_width/2 and 
                                abs(gaze_y - button_y) <= button_height/2):
                                in_button_points += 1
                    
                    # 计算命中率 - 修改计算方式
                    if in_range_points > 0:
                        gaze_hit_rate = in_button_points / in_range_points
                    else:
                        # 如果没有在范围内的点，检查是否有任何注视点
                        if len(sampled_data) > 0:
                            # 如果有注视点但都不在范围内，说明注视点可能偏离太远
                            gaze_hit_rate = 0
                        else:
                            # 如果完全没有注视点数据，使用默认值
                            gaze_hit_rate = 0
                    
                    gaze_hit_rate_by_round.append(gaze_hit_rate)
                else:
                    gaze_hit_rate_by_round.append(0)
            else:
                # 如果该轮没有数据，添加默认值
                mae_by_round.append(0)
                gaze_hit_rate_by_round.append(0)
        
        # 绘制趋势图
        plt.figure(figsize=(15, 5))
        
        # MAE趋势
        plt.subplot(121)
        plt.plot(range(1, 13), mae_by_round, 'b-o')  # 12个round
        plt.title('MAE Trend')
        plt.xlabel('Round')
        plt.ylabel('MAE (pixels)')
        
        # 注视点命中率趋势
        plt.subplot(122)
        plt.plot(range(1, 13), gaze_hit_rate_by_round, 'g-o')  # 12个round
        plt.title('Gaze Hit Rate Trend')
        plt.xlabel('Round')
        plt.ylabel('Hit Rate')
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/trend_analysis.png")
        plt.close()
        
        # 生成文本报告
        with open(f"{self.experiment_dir}/experiment_report.txt", 'w') as f:
            f.write(f"Experiment Report for {self.subject_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Overall Statistics:\n")
            f.write(f"Total Rounds: {len(mae_by_round)}\n")
            f.write(f"Average MAE: {np.mean(mae_by_round):.2f} pixels\n")
            f.write(f"Average Gaze Hit Rate: {np.mean(gaze_hit_rate_by_round):.2%}\n\n")
            
            f.write("Round-by-Round Analysis:\n")
            for i in range(len(mae_by_round)):
                f.write(f"\nRound {i+1}:\n")
                f.write(f"MAE: {mae_by_round[i]:.2f} pixels\n")
                f.write(f"Gaze Hit Rate: {gaze_hit_rate_by_round[i]:.2%}\n")

def load_experiment_buttons():
    """加载实验按钮配置"""
    try:
        with open(EXPERIMENT_BUTTONS_FILE, "r", encoding='utf-8') as f:
            return json.load(f)["buttons"]
    except Exception as e:
        print(f"加载实验按钮配置失败: {e}")
        return []

def is_point_in_button(point, button):
    """判断点是否在按钮区域内"""
    return (button["x"] <= point[0] <= button["x"] + button["width"] and
            button["y"] <= point[1] <= button["y"] + button["height"])

def collect_implicit_calibration_data(gaze, button):
    """收集隐式校准数据并更新校准文件"""
    samples = []
    start_time = time.time()
    
    while time.time() - start_time < 2.0:  # 采样2秒
        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()
        
        if hr is not None and vr is not None:
            samples.append((hr, vr))
        
        time.sleep(0.1)
    
    if len(samples) > 0:
        # 计算平均值
        avg_hr = sum(s[0] for s in samples) / len(samples)
        avg_vr = sum(s[1] for s in samples) / len(samples)
        
        button_center_x = button["x"] + button["width"] // 2
        button_center_y = button["y"] + button["height"] // 2
        
        # 读取现有校准数据
        try:
            with open(CALIBRATION_FILE, "r") as f:
                calibration_data = json.load(f)
        except FileNotFoundError:
            calibration_data = []
        
        # 添加新的校准点
        calibration_data.append({
            "hr": avg_hr,
            "vr": avg_vr,
            "x": button_center_x,
            "y": button_center_y,
            "button_name": button["name"]
        })
        
        # 保存更新后的校准数据
        try:
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(calibration_data, f, indent=4)
            print(f"已更新校准文件，添加了按钮 {button['name']} 的校准数据")
        except Exception as e:
            print(f"保存校准数据出错: {e}")

class ExperimentUI:
    def __init__(self, experiment_data):
        self.experiment_data = experiment_data
        self.buttons = self.calculate_button_positions()
        self.current_button_index = 0
        self.remaining_buttons = list(range(len(self.buttons)))
        random.shuffle(self.remaining_buttons)
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("Gaze Tracking Experiment")
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.configure(bg='white')
        
        # 创建Canvas
        self.canvas = tk.Canvas(self.root, width=screen_width, height=screen_height,
                              bg="white", highlightthickness=0)
        self.canvas.pack()
        
        # 创建按钮显示
        self.button_rectangles = []
        for button in self.buttons:
            rect = self.canvas.create_rectangle(
                button["x"], button["y"],
                button["x"] + button["width"],
                button["y"] + button["height"],
                fill="gray", outline="black"
            )
            self.button_rectangles.append(rect)
        
        # 创建视线位置指示点
        self.dot_radius = 10
        self.dot = self.canvas.create_oval(0, 0, self.dot_radius*2, self.dot_radius*2,
                                         fill="blue", outline="blue")
        
        self.text_item = self.canvas.create_text(50, 50, text="Gaze: ( , )",
                                               fill="black", font=("Arial", 20, "bold"))
        
        # 高亮当前按钮
        self.highlight_current_button()
    
    def calculate_button_positions(self):
        """根据屏幕大小计算按钮位置"""
        # 设置按钮大小
        button_width = 100
        button_height = 100
        
        # 计算边距和间距
        margin = 50
        spacing = 200  # 进一步增加间距
        
        # 设置三排，每排四个按钮
        buttons_per_row = 4
        rows = 3
        
        # 计算起始位置
        total_width = buttons_per_row * button_width + (buttons_per_row - 1) * spacing
        total_height = rows * button_height + (rows - 1) * spacing
        
        start_x = (screen_width - total_width) // 2
        start_y = (screen_height - total_height) // 2
        
        buttons = []
        button_id = 1
        
        for row in range(rows):
            for col in range(buttons_per_row):
                if button_id > 12:  # 总共12个按钮
                    break
                    
                x = start_x + col * (button_width + spacing)
                y = start_y + row * (button_height + spacing)
                
                buttons.append({
                    "id": button_id,
                    "name": f"Button {button_id}",
                    "x": x,
                    "y": y,
                    "width": button_width,
                    "height": button_height
                })
                button_id += 1
        
        return buttons
    
    def highlight_current_button(self):
        """高亮当前目标按钮"""
        # 重置所有按钮颜色
        for rect in self.button_rectangles:
            self.canvas.itemconfig(rect, fill="gray")
        
        # 高亮当前按钮
        if self.current_button_index < len(self.remaining_buttons):
            current_button_id = self.remaining_buttons[self.current_button_index]
            self.canvas.itemconfig(self.button_rectangles[current_button_id], fill="yellow")
    
    def update_gaze_position(self, x, y):
        """更新视线位置"""
        self.canvas.coords(self.dot,
                         x - self.dot_radius,
                         y - self.dot_radius,
                         x + self.dot_radius,
                         y + self.dot_radius)
        self.canvas.itemconfig(self.text_item, text=f"Gaze: ({x}, {y})")
    
    def next_button(self):
        """进入下一个按钮（下一个round）"""
        self.current_button_index += 1
        if self.current_button_index < len(self.remaining_buttons):
            self.highlight_current_button()
            return True
        return False
    
    def get_current_button(self):
        """获取当前目标按钮"""
        if self.current_button_index < len(self.remaining_buttons):
            return self.buttons[self.remaining_buttons[self.current_button_index]]
        return None

def main():
    """主程序"""
    # 初始化实验数据
    experiment_data = ExperimentData(SUBJECT_NAME,USE_IMPLICIT_CALIBRATION)
    experiment_data.start_time = time.time()
    
    # 初始化眼动跟踪
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    
    # 创建UI
    ui = ExperimentUI(experiment_data)
    
    # 创建卡尔曼滤波器
    kf = create_kalman_filter_acc()
    initialized = False
    last_filtered = None
    
    # 用于移动平均的点列表
    recent_points = []
    last_stable_point = None
    last_update_time = time.time()
    
    # 停留点击相关状态
    dwell_center = None
    dwell_start_time = 0.0
    in_cooldown = False
    cooldown_start = 0.0
    
    while True:
        # 读取摄像头帧
        ret, frame = webcam.read()
        if not ret:
            break
        
        # 分析眼动
        gaze.refresh(frame)
        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()
        
        if hr is not None and vr is not None:
            # 加载最新的校准数据
            calibration_data = load_calibration()
            gaze_x, gaze_y = get_gaze_coordinates(hr, vr, calibration_data)
            
            if gaze_x is not None and gaze_y is not None:
                # 记录原始数据
                current_button = ui.get_current_button()
                if current_button:
                    experiment_data.add_raw_data(gaze_x, gaze_y, current_button)
                
                # 初始化卡尔曼滤波
                if not initialized:
                    kf.x = np.array([gaze_x, gaze_y, 0, 0, 0, 0])
                    last_filtered = (gaze_x, gaze_y)
                    initialized = True
                
                # 卡尔曼滤波
                kf.predict()
                kf.update([gaze_x, gaze_y])
                kalman_point = (int(kf.x[0]), int(kf.x[1]))
                
                # 计算速度
                current_velocity = np.sqrt(kf.x[2]**2 + kf.x[3]**2)
                
                # 低通滤波
                filtered_point = low_pass_filter(kalman_point, last_filtered,
                                               alpha_x=0.2, alpha_y=0.1)
                
                # 移动平均
                recent_points.append(filtered_point)
                if len(recent_points) > MOVING_AVERAGE_WINDOW:
                    recent_points.pop(0)
                averaged_point = moving_average(recent_points)
                
                # 防抖动处理
                current_time = time.time()
                dt = current_time - last_update_time
                
                if current_velocity < MIN_VELOCITY_THRESHOLD:
                    if last_stable_point is not None:
                        averaged_point = last_stable_point
                elif is_in_dead_zone(averaged_point, last_filtered):
                    averaged_point = last_filtered
                
                # 更新状态
                last_filtered = averaged_point
                last_stable_point = averaged_point
                last_update_time = current_time
                
                # 更新UI（显示处理后的位置）
                ui.update_gaze_position(averaged_point[0], averaged_point[1])
                
                # 停留点击逻辑
                now = time.time()
                
                if in_cooldown:
                    if now - cooldown_start >= COOLDOWN_AFTER_CLICK:
                        in_cooldown = False
                else:
                    if dwell_center is None:
                        dwell_center = averaged_point
                        dwell_start_time = now
                    else:
                        dist = distance(averaged_point, dwell_center)
                        if dist <= DISTANCE_THRESHOLD:
                            dwell_duration = now - dwell_start_time
                            if dwell_duration >= DURATION_THRESHOLD:
                                # 触发点击
                                current_button = ui.get_current_button()
                                if current_button:
                                    # 记录点击数据
                                    hit = is_point_in_button(dwell_center, current_button)
                                    experiment_data.add_processed_data(
                                        dwell_center[0], dwell_center[1],
                                        current_button, hit
                                    )
                                    
                                    # 收集隐式校准数据并更新校准文件
                                    collect_implicit_calibration_data(gaze, current_button)
                                    
                                    # 执行点击
                                    pyautogui.click(dwell_center[0], dwell_center[1])
                                    
                                    # 进入下一个按钮（下一个round）
                                    if not ui.next_button():
                                        # 实验结束
                                        experiment_data.save_data()
                                        return
                                    
                                    # 进入下一轮
                                    experiment_data.next_round()
                                    
                                    # 进入冷却期
                                    in_cooldown = True
                                    cooldown_start = now
                                    dwell_center = None
                        else:
                            dwell_center = averaged_point
                            dwell_start_time = now
        
        # 更新UI
        ui.root.update()
        
        # 检查退出
        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # ESC
            break
    
    # 清理
    webcam.release()
    cv2.destroyAllWindows()
    ui.root.destroy()

if __name__ == "__main__":
    main() 