import cv2
import numpy as np
import pyautogui
import json
import time
import tkinter as tk
import webbrowser
import os
from gaze_tracking import GazeTracking
from filterpy.kalman import KalmanFilter

# 可调参数
PRIOR_CALIBRATION_FILE = "calibration-5.json"  # 先验校准数据
CALIBRATION_FILE = "calibration_implicit.json"  # 隐式校准数据
DURATION_THRESHOLD = 2.0       # 按钮停留阈值(秒)
COOLDOWN_AFTER_CLICK = 1.0    # 点击后冷却时长(秒)
VERTICAL_THRESHOLD = 20        # 垂直方向移动阈值，用于检测换行
MIN_POINTS_PER_LINE = 2        # 每行最少需要的点数
MAX_CALIBRATION_POINTS = 100   # 最大保存的校准点数

# 定义页面区域
class PageRegions:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        
    def get_home_page_regions(self):
        """首页区域定义"""
        return {
            'text': {
                'x1': self.screen_width * 0.2,
                'y1': self.screen_height * 0.3,
                'x2': self.screen_width * 0.8,
                'y2': self.screen_height * 0.5
            },
            'button': {
                'x1': self.screen_width * 0.45,
                'y1': self.screen_height * 0.6,
                'x2': self.screen_width * 0.55,
                'y2': self.screen_height * 0.65
            }
        }
        
    def get_content_page_regions(self):
        """内容页区域定义"""
        regions = []
        # 每个section的区域
        for i in range(8):  # tutorial.html中有8个section
            y_offset = i * (self.screen_height * 0.4)  # 每个section的高度为屏幕高度的40%
            regions.append({
                'text': {
                    'x1': self.screen_width * 0.2,
                    'y1': y_offset + self.screen_height * 0.1,
                    'x2': self.screen_width * 0.8,
                    'y2': y_offset + self.screen_height * 0.3
                },
                'button': None  # 内容页没有按钮
            })
        return regions
        
    def get_complete_page_regions(self):
        """完成页区域定义"""
        return {
            'text': {
                'x1': self.screen_width * 0.7,
                'y1': self.screen_height * 0.1,
                'x2': self.screen_width * 0.9,
                'y2': self.screen_height * 0.3
            },
            'button': None  # 完成页没有按钮
        }

class TutorialCalibrator:
    def __init__(self):
        self.calibration_data = self.load_prior_calibration()
        self.reading_tracker = ReadingTracker()
        self.kf = self.create_kalman_filter()
        self.initialized = False
        self.last_filtered = None
        
        # 区域管理
        self.regions = PageRegions()
        self.current_page = 'home'
        self.current_section = 0
        
        # 按钮动画相关
        self.button_progress = 0.0
        self.button_start_time = 0.0
        self.is_button_hovered = False
        
        # 创建透明窗口
        self.root = tk.Tk()
        self.root.title("眼动控制教程 - 实时校准")
        screen_width, screen_height = pyautogui.size()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.configure(bg='black')
        self.root.wm_attributes('-transparentcolor', 'black')
        self.root.wm_attributes('-topmost', 1)
        
        # 创建画布
        self.canvas = tk.Canvas(self.root, width=screen_width, height=screen_height,
                              bg="black", highlightthickness=0)
        self.canvas.pack()
        
        # 创建视线指示器
        self.dot_radius = 10
        self.dot = self.canvas.create_oval(0, 0, self.dot_radius*2, self.dot_radius*2,
                                         fill="red", outline="red")
        self.text_item = self.canvas.create_text(50, 50, text="视线: ( , )",
                                               fill="red", font=("Arial", 20, "bold"))
        
        # 创建状态显示
        self.status_text = self.canvas.create_text(50, 100, 
                                                 text="校准点数: 0",
                                                 fill="red", font=("Arial", 20, "bold"))
        
        # 创建按钮动画
        self.button_circle = self.canvas.create_oval(0, 0, 0, 0,
                                                   outline="blue", width=2)
        self.button_progress_arc = self.canvas.create_arc(0, 0, 0, 0,
                                                        start=0, extent=0,
                                                        outline="blue", width=2)
        
    def load_prior_calibration(self):
        """加载先验校准数据"""
        try:
            with open(PRIOR_CALIBRATION_FILE, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("先验校准文件未找到。请先运行 calibration-5.py 进行校准。")
            return None
            
    def create_kalman_filter(self):
        """创建卡尔曼滤波器"""
        kf = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        kf.P *= 10.
        kf.R *= 20.
        kf.Q *= 0.01
        return kf
        
    def get_gaze_coordinates(self, hr, vr):
        """根据校准数据，将眼动比例转换为屏幕坐标"""
        if not self.calibration_data:
            return None, None

        # 使用加权平均计算坐标
        x_vals = np.array([p["hr"] for p in self.calibration_data])
        y_vals = np.array([p["vr"] for p in self.calibration_data])
        screen_x_vals = np.array([p["x"] for p in self.calibration_data])
        screen_y_vals = np.array([p["y"] for p in self.calibration_data])

        # 计算与当前点的距离
        distances = np.sqrt((x_vals - hr)**2 + (y_vals - vr)**2)
        weights = np.exp(-distances / 0.1)  # 使用指数衰减作为权重
        weights = weights / np.sum(weights)  # 归一化权重

        # 计算加权平均坐标
        gaze_x = int(np.sum(weights * screen_x_vals))
        gaze_y = int(np.sum(weights * screen_y_vals))
        
        # 添加边界限制
        gaze_x = max(0, min(gaze_x, self.regions.screen_width))
        gaze_y = max(0, min(gaze_y, self.regions.screen_height))
        
        return gaze_x, gaze_y
        
    def distance(self, p1, p2):
        """计算两点欧几里得距离"""
        return np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1]])
        
    def update_calibration(self):
        """更新校准数据"""
        if self.reading_tracker.calibration_points:
            # 合并先验数据和新的校准点
            self.calibration_data.extend(self.reading_tracker.calibration_points)
            # 保持最大点数
            if len(self.calibration_data) > MAX_CALIBRATION_POINTS:
                self.calibration_data = self.calibration_data[-MAX_CALIBRATION_POINTS:]
            # 保存到文件
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(self.calibration_data, f, indent=4)
            # 清空临时校准点
            self.reading_tracker.calibration_points = []
            
    def is_point_in_region(self, point, region):
        """检查点是否在指定区域内"""
        if region is None:
            return False
        return (region['x1'] <= point[0] <= region['x2'] and 
                region['y1'] <= point[1] <= region['y2'])
        
    def get_current_regions(self):
        """获取当前页面的区域定义"""
        if self.current_page == 'home':
            return self.regions.get_home_page_regions()
        elif self.current_page == 'content':
            regions = self.regions.get_content_page_regions()
            return regions[self.current_section]
        else:  # complete
            return self.regions.get_complete_page_regions()
            
    def update_button_animation(self, point, regions):
        """更新按钮动画"""
        if regions['button'] is None:
            self.is_button_hovered = False
            self.button_progress = 0.0
            return
            
        button = regions['button']
        if self.is_point_in_region(point, button):
            if not self.is_button_hovered:
                self.is_button_hovered = True
                self.button_start_time = time.time()
                # 设置按钮圆圈位置
                center_x = (button['x1'] + button['x2']) / 2
                center_y = (button['y1'] + button['y2']) / 2
                radius = min(button['x2'] - button['x1'], button['y2'] - button['y1']) / 4
                self.canvas.coords(self.button_circle,
                                 center_x - radius, center_y - radius,
                                 center_x + radius, center_y + radius)
                self.canvas.coords(self.button_progress_arc,
                                 center_x - radius, center_y - radius,
                                 center_x + radius, center_y + radius)
            
            # 更新进度
            progress = min(1.0, (time.time() - self.button_start_time) / DURATION_THRESHOLD)
            self.button_progress = progress
            self.canvas.itemconfig(self.button_progress_arc,
                                 start=90, extent=-360 * progress)
            
            # 如果进度达到1，触发点击
            if progress >= 1.0:
                pyautogui.click(center_x, center_y)
                self.update_calibration()
                self.is_button_hovered = False
                self.button_progress = 0.0
        else:
            self.is_button_hovered = False
            self.button_progress = 0.0
            
    def update_text_calibration(self, point, regions):
        """更新文本校准"""
        if regions['text'] is None:
            return
            
        if self.is_point_in_region(point, regions['text']):
            # 在文本区域内，进行校准
            # 将屏幕坐标转换回 hr 和 vr
            hr = point[0] / self.regions.screen_width
            vr = point[1] / self.regions.screen_height
            # 直接使用 point[0] 和 point[1] 作为屏幕坐标
            self.reading_tracker.track_gaze(hr, vr, point[0], point[1], time.time())
        else:
            # 离开文本区域，处理当前行的数据
            self.reading_tracker.process_line()
            
    def run(self):
        """运行主循环"""
        # 启动tutorial.html
        webbrowser.open('file://' + os.path.realpath('tutorial.html'))
        
        # 初始化眼动跟踪和摄像头
        gaze = GazeTracking()
        webcam = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = webcam.read()
                if not ret:
                    break
                    
                # 更新眼动数据
                gaze.refresh(frame)
                hr = gaze.horizontal_ratio()
                vr = gaze.vertical_ratio()
                
                if hr is not None and vr is not None:
                    # 获取屏幕坐标
                    gaze_x, gaze_y = self.get_gaze_coordinates(hr, vr)
                    
                    if gaze_x is not None and gaze_y is not None:
                        # 初始化卡尔曼滤波
                        if not self.initialized:
                            self.kf.x = np.array([gaze_x, gaze_y, 0, 0, 0, 0])
                            self.last_filtered = (gaze_x, gaze_y)
                            self.initialized = True
                            
                        # 卡尔曼滤波
                        self.kf.predict()
                        self.kf.update([gaze_x, gaze_y])
                        kalman_point = (int(self.kf.x[0]), int(self.kf.x[1]))
                        
                        # 更新视线指示器
                        self.canvas.coords(self.dot,
                                        kalman_point[0] - self.dot_radius,
                                        kalman_point[1] - self.dot_radius,
                                        kalman_point[0] + self.dot_radius,
                                        kalman_point[1] + self.dot_radius)
                        info = f"视线: ({kalman_point[0]}, {kalman_point[1]})"
                        self.canvas.itemconfig(self.text_item, text=info)
                        
                        # 更新状态显示
                        status = f"校准点数: {len(self.calibration_data)}"
                        self.canvas.itemconfig(self.status_text, text=status)
                        
                        # 获取当前区域
                        regions = self.get_current_regions()
                        
                        # 更新按钮动画
                        self.update_button_animation(kalman_point, regions)
                        
                        # 更新文本校准
                        self.update_text_calibration(kalman_point, regions)
                        
                # 更新窗口
                self.root.update()
                
                # 按ESC退出
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
        finally:
            # 保存最终校准数据
            self.update_calibration()
            webcam.release()
            cv2.destroyAllWindows()
            self.root.destroy()

class ReadingTracker:
    def __init__(self):
        self.current_line_start = None
        self.current_line_start_time = None
        self.last_position = None
        self.line_points = []  # 存储当前行的点
        self.calibration_points = []  # 存储所有校准点
        self.kf = self.create_kalman_filter()
        self.initialized = False
        
    def create_kalman_filter(self):
        """创建卡尔曼滤波器"""
        kf = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        kf.P *= 10.
        kf.R *= 20.
        kf.Q *= 0.01
        return kf

    def track_gaze(self, hr, vr, x, y, current_time):
        """跟踪视线位置"""
        if not self.initialized:
            self.kf.x = np.array([x, y, 0, 0, 0, 0])
            self.initialized = True
            
        # 卡尔曼滤波
        self.kf.predict()
        self.kf.update([x, y])
        filtered_x = int(self.kf.x[0])
        filtered_y = int(self.kf.x[1])
        
        current_position = (filtered_x, filtered_y)
        
        if self.current_line_start is None:
            # 开始新的一行
            self.current_line_start = (hr, vr, filtered_x, filtered_y)
            self.current_line_start_time = current_time
            self.line_points = [(hr, vr, filtered_x, filtered_y)]
            
        # 检查是否有垂直方向的大转折
        if self.last_position and abs(filtered_y - self.last_position[1]) > VERTICAL_THRESHOLD:
            # 用户换行了，处理上一行的数据
            self.process_line()
            # 开始新的一行
            self.current_line_start = (hr, vr, filtered_x, filtered_y)
            self.current_line_start_time = current_time
            self.line_points = [(hr, vr, filtered_x, filtered_y)]
            
        self.line_points.append((hr, vr, filtered_x, filtered_y))
        self.last_position = current_position
        
    def process_line(self):
        """处理当前行的数据"""
        if len(self.line_points) >= MIN_POINTS_PER_LINE:
            # 获取开始点和结束点
            start_point = self.line_points[0]
            end_point = self.line_points[-1]
            
            # 计算中间点（时间中点）
            mid_time = (self.current_line_start_time + time.time()) / 2
            mid_point = self.find_point_at_time(mid_time)
            
            if mid_point:
                # 添加校准点
                self.calibration_points.extend([
                    {"hr": start_point[0], "vr": start_point[1], "x": start_point[2], "y": start_point[3]},
                    {"hr": mid_point[0], "vr": mid_point[1], "x": mid_point[2], "y": mid_point[3]},
                    {"hr": end_point[0], "vr": end_point[1], "x": end_point[2], "y": end_point[3]}
                ])
                
                # 保持最大点数
                if len(self.calibration_points) > MAX_CALIBRATION_POINTS:
                    self.calibration_points = self.calibration_points[-MAX_CALIBRATION_POINTS:]
    
    def find_point_at_time(self, target_time):
        """在轨迹中找到指定时间点的位置"""
        if not self.line_points:
            return None
            
        # 简单线性插值
        start_time = self.current_line_start_time
        end_time = time.time()
        
        if target_time <= start_time:
            return self.line_points[0]
        if target_time >= end_time:
            return self.line_points[-1]
            
        t = (target_time - start_time) / (end_time - start_time)
        start_point = self.line_points[0]
        end_point = self.line_points[-1]
        
        return (
            start_point[0] + (end_point[0] - start_point[0]) * t,
            start_point[1] + (end_point[1] - start_point[1]) * t,
            start_point[2] + (end_point[2] - start_point[2]) * t,
            start_point[3] + (end_point[3] - start_point[3]) * t
        )

if __name__ == "__main__":
    calibrator = TutorialCalibrator()
    calibrator.run() 