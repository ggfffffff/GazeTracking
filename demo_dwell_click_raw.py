import cv2
import numpy as np
import pyautogui
import json
import time
import tkinter as tk
from gaze_tracking import GazeTracking

# ==============================
# 可调参数
DURATION_THRESHOLD = 1.0       # 停留阈值(秒)——在同一区域停留时间达到此值才触发点击
DISTANCE_THRESHOLD = 100        # 稳定范围(像素)——视线在此范围内认为是"停留"
COOLDOWN_AFTER_CLICK = 1.0     # 点击后冷却时长(秒)，防止连点

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

    poly_x = np.polyfit(x_vals, screen_x_vals, 1)
    poly_y = np.polyfit(y_vals, screen_y_vals, 1)

    gaze_x = int(np.polyval(poly_x, hr))
    gaze_y = int(np.polyval(poly_y, vr))
    return gaze_x, gaze_y

def distance(p1, p2):
    """计算两点欧几里得距离"""
    return np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1]])

def main():
    """主程序：读取摄像头、计算 gaze 位置并在透明窗口上显示 + 停留点击"""
    calibration_data = load_calibration()
    if not calibration_data:
        print("No calibration data. Please run calibration first.")
        return

    # 初始化眼动跟踪和摄像头
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    # ----------- 创建透明 Tkinter 窗口 -----------
    root = tk.Tk()
    root.title("Gaze Transparent Demo - Raw Version")
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
                # 直接使用原始坐标，不做任何滤波
                current_point = (gaze_x, gaze_y)

                # 在透明窗口上更新红点位置
                canvas.coords(dot,
                              current_point[0] - dot_radius,
                              current_point[1] - dot_radius,
                              current_point[0] + dot_radius,
                              current_point[1] + dot_radius)
                info = f"Gaze: ({current_point[0]}, {current_point[1]})"
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
                        dwell_center = current_point
                        dwell_start_time = now
                    else:
                        dist = distance(current_point, dwell_center)
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
                            dwell_center = current_point
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