import cv2
import numpy as np
import pyautogui
import json
import time
import tkinter as tk
import webbrowser
from gaze_tracking import GazeTracking
from filterpy.kalman import KalmanFilter
import os

# ==============================
# 可调参数
DURATION_THRESHOLD = 1.0       # 停留阈值(秒)——在同一区域停留时间达到此值才触发点击
DISTANCE_THRESHOLD = 100       # 稳定范围(像素)——视线在此范围内认为是"停留"
COOLDOWN_AFTER_CLICK = 1.0     # 点击后冷却时长(秒)，防止连点
MOVING_AVERAGE_WINDOW = 5      # 移动平均窗口大小
DEAD_ZONE = 5                  # 死区范围（像素），在此范围内的移动视为静止
MIN_VELOCITY_THRESHOLD = 0.5   # 最小速度阈值，低于此速度视为静止
Y_OFFSET = -10                 # Y轴偏移量，用于微调点击位置

# ==============================
# 校准文件路径
CALIBRATION_FILE = "calibration-5.json"
# 获取屏幕分辨率
screen_width, screen_height = pyautogui.size()

def load_calibration():
    """加载校准数据"""
    try:
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Calibration file not found. Please run calibration.py first.")
        return None

def get_gaze_coordinates(hr, vr, calibration_data):
    """
    根据校准数据，将 gaze tracking 的水平 & 垂直比例转换为屏幕坐标
    """
    if not calibration_data:
        return None, None

    x_vals = np.array([p["hr"] for p in calibration_data])
    y_vals = np.array([p["vr"] for p in calibration_data])
    screen_x_vals = np.array([p["x"] for p in calibration_data])
    screen_y_vals = np.array([p["y"] for p in calibration_data])

    # 使用二次多项式拟合，提高精度
    poly_x = np.polyfit(x_vals, screen_x_vals, 2)
    poly_y = np.polyfit(y_vals, screen_y_vals, 2)

    gaze_x = int(np.polyval(poly_x, hr))
    gaze_y = int(np.polyval(poly_y, vr)) + Y_OFFSET  # 添加Y轴偏移
    
    # 添加边界限制
    gaze_x = max(0, min(gaze_x, screen_width))
    gaze_y = max(0, min(gaze_y, screen_height))
    
    return gaze_x, gaze_y

def create_kalman_filter_acc():
    """
    创建一个基于常加速模型的卡尔曼滤波器，状态向量为 [x, y, vx, vy, ax, ay]
    """
    kf = KalmanFilter(dim_x=6, dim_z=2)
    dt = 1.0
    kf.F = np.array([
        [1, 0, dt, 0, 0.5*dt*dt, 0],
        [0, 1, 0, dt, 0, 0.5*dt*dt],
        [0, 0, 1, 0, dt,       0],
        [0, 0, 0, 1, 0,       dt],
        [0, 0, 0, 0, 1,        0],
        [0, 0, 0, 0, 0,        1]
    ])
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ])
    kf.P *= 10.
    kf.R *= 20.   # 增大测量噪声，使滤波器更平滑
    kf.Q *= 0.01
    return kf

def low_pass_filter(new_point, last_filtered, alpha_x=0.2, alpha_y=0.3):
    """
    对新坐标进行指数加权低通滤波，分别对 x 和 y 方向使用不同的平滑系数
    alpha_x: x 方向平滑系数
    alpha_y: y 方向平滑系数（设得更小，则 y 方向平滑力度更强）
    """
    if last_filtered is None:
        return new_point
    filtered_x = int(alpha_x * new_point[0] + (1 - alpha_x) * last_filtered[0])
    filtered_y = int(alpha_y * new_point[1] + (1 - alpha_y) * last_filtered[1])
    return (filtered_x, filtered_y)

def distance(p1, p2):
    """计算两点欧几里得距离"""
    return np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1]])

def moving_average(points):
    """
    计算移动平均点
    points: 最近的点列表
    """
    if len(points) < 2:
        return points[-1]
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    return (int(x_sum / len(points)), int(y_sum / len(points)))

def is_in_dead_zone(point1, point2):
    """
    判断两点是否在死区内
    """
    return abs(point1[0] - point2[0]) <= DEAD_ZONE and abs(point1[1] - point2[1]) <= DEAD_ZONE

def open_webpage():
    """打开网页"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建tutorial.html的完整路径
    html_path = os.path.join(current_dir, "tutorial.html")
    # 使用默认浏览器打开网页
    webbrowser.open('file://' + html_path)

def main():
    """主程序：打开网页并启动眼动识别"""
    # 首先打开网页
    open_webpage()
    
    # 等待1秒，让浏览器有时间打开
    time.sleep(1)
    
    calibration_data = load_calibration()
    if not calibration_data:
        print("No calibration data. Please run calibration first.")
        return

    # 初始化眼动跟踪和摄像头
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    # 创建卡尔曼滤波器
    kf = create_kalman_filter_acc()
    initialized = False
    last_filtered = None
    delta_threshold = 5
    
    # 新增：用于移动平均的点列表
    recent_points = []
    last_stable_point = None
    last_update_time = time.time()

    # ----------- 创建透明 Tkinter 窗口 -----------
    root = tk.Tk()
    root.title("Gaze Tracking Overlay")
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    # 透明背景
    root.configure(bg='black')
    root.wm_attributes('-transparentcolor', 'black')
    root.wm_attributes('-topmost', 1)  # 保持窗口最前

    # 在 Tkinter 窗口上创建 Canvas
    canvas = tk.Canvas(root, width=screen_width, height=screen_height,
                       bg="black", highlightthickness=0)
    canvas.pack()

    # 在 Canvas 上创建一个红点和文本，用于显示视线坐标
    dot_radius = 10
    dot = canvas.create_oval(0, 0, dot_radius*2, dot_radius*2,
                             fill="red", outline="red")
    text_item = canvas.create_text(50, 50, text="Gaze: ( , )",
                                   fill="red", font=("Arial", 20, "bold"))

    # =============== 停留点击相关状态 ===============
    dwell_center = None         # 当前停留位置中心
    dwell_start_time = 0.0      # 在该位置开始停留的时刻
    in_cooldown = False
    cooldown_start = 0.0

    while True:
        # 读取摄像头帧
        ret, frame = webcam.read()
        if not ret:
            break

        # 用 GazeTracking 分析帧
        gaze.refresh(frame)
        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()

        if hr is not None and vr is not None:
            gaze_x, gaze_y = get_gaze_coordinates(hr, vr, calibration_data)
            if gaze_x is not None and gaze_y is not None:
                # 初始化卡尔曼滤波状态
                if not initialized:
                    kf.x = np.array([gaze_x, gaze_y, 0, 0, 0, 0])
                    last_filtered = (gaze_x, gaze_y)
                    initialized = True

                # 卡尔曼预测 & 更新
                kf.predict()
                kf.update([gaze_x, gaze_y])
                kalman_point = (int(kf.x[0]), int(kf.x[1]))

                # 计算当前速度
                current_velocity = np.sqrt(kf.x[2]**2 + kf.x[3]**2)
                
                # 低通滤波
                filtered_point = low_pass_filter(kalman_point, last_filtered,
                                                 alpha_x=0.2, alpha_y=0.1)
                
                # 添加移动平均
                recent_points.append(filtered_point)
                if len(recent_points) > MOVING_AVERAGE_WINDOW:
                    recent_points.pop(0)
                averaged_point = moving_average(recent_points)
                
                # 防抖动处理
                current_time = time.time()
                dt = current_time - last_update_time
                
                if current_velocity < MIN_VELOCITY_THRESHOLD:
                    # 速度很小，保持静止
                    if last_stable_point is not None:
                        averaged_point = last_stable_point
                elif is_in_dead_zone(averaged_point, last_filtered):
                    # 在死区内，保持静止
                    averaged_point = last_filtered
                
                # 更新状态
                last_filtered = averaged_point
                last_stable_point = averaged_point
                last_update_time = current_time

                # 在透明窗口上更新红点位置
                canvas.coords(dot,
                              averaged_point[0] - dot_radius,
                              averaged_point[1] - dot_radius,
                              averaged_point[0] + dot_radius,
                              averaged_point[1] + dot_radius)
                info = f"Gaze: ({averaged_point[0]}, {averaged_point[1]})"
                canvas.itemconfig(text_item, text=info)

                # ========== 停留点击逻辑 ==========
                now = time.time()

                if in_cooldown:
                    # 点击冷却期中，不检查点击
                    if now - cooldown_start >= COOLDOWN_AFTER_CLICK:
                        in_cooldown = False
                else:
                    # 不在冷却期，检查视线是否在小范围内停留
                    if dwell_center is None:
                        # 还没记录任何停留位置，则记录当前坐标
                        dwell_center = averaged_point
                        dwell_start_time = now
                    else:
                        dist = distance(averaged_point, dwell_center)
                        if dist <= DISTANCE_THRESHOLD:
                            # 继续停留在同一小范围
                            dwell_duration = now - dwell_start_time
                            if dwell_duration >= DURATION_THRESHOLD:
                                # 触发点击
                                print("Dwell Click at", dwell_center)
                                pyautogui.click(dwell_center[0], dwell_center[1])

                                # 进入冷却期
                                in_cooldown = True
                                cooldown_start = now

                                # 重置停留
                                dwell_center = None
                        else:
                            # 超出范围，重新记录新的停留中心
                            dwell_center = averaged_point
                            dwell_start_time = now

        # 更新 Tkinter 窗口
        root.update()

        # 按 ESC 键退出
        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # ESC
            break

    # 退出时清理
    webcam.release()
    cv2.destroyAllWindows()
    root.destroy()

if __name__ == "__main__":
    main() 