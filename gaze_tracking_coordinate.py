import cv2
import numpy as np
import json
import tkinter as tk
from gaze_tracking import GazeTracking
import pyautogui

# 校准文件路径
CALIBRATION_FILE = "calibration-5.json"
# 获取屏幕分辨率
screen_width, screen_height = pyautogui.size()

# 可调参数
Y_OFFSET = -10  # Y轴偏移量，用于微调点击位置

def load_calibration():
    """加载校准数据"""
    try:
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("校准文件未找到。请先运行 calibration.py 进行校准。")
        return None

def get_gaze_coordinates(hr, vr, calibration_data):
    """
    优化版本：使用二次多项式拟合，提高精度
    分别对水平和垂直方向进行独立拟合
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

def main():
    """主程序：读取摄像头、计算 gaze 位置并在透明窗口上显示"""
    calibration_data = load_calibration()
    if not calibration_data:
        print("No calibration data. Please run calibration first.")
        return

    # 初始化眼动跟踪和摄像头
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    # ----------- 创建透明 Tkinter 窗口 -----------
    root = tk.Tk()
    root.title("Gaze Tracking Coordinate")
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.configure(bg='black')
    root.wm_attributes('-transparentcolor', 'black')
    root.wm_attributes('-topmost', 1)

    canvas = tk.Canvas(root, width=screen_width, height=screen_height,
                      bg="black", highlightthickness=0)
    canvas.pack()

    dot_radius = 10
    dot = canvas.create_oval(0, 0, dot_radius*2, dot_radius*2,
                           fill="red", outline="red")
    text_item = canvas.create_text(50, 50, text="Gaze: ( , )",
                                 fill="red", font=("Arial", 20, "bold"))

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        gaze.refresh(frame)
        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()

        if hr is not None and vr is not None:
            gaze_x, gaze_y = get_gaze_coordinates(hr, vr, calibration_data)
            if gaze_x is not None and gaze_y is not None:
                canvas.coords(dot,
                            gaze_x - dot_radius,
                            gaze_y - dot_radius,
                            gaze_x + dot_radius,
                            gaze_y + dot_radius)
                info = f"Gaze: ({gaze_x}, {gaze_y})"
                canvas.itemconfig(text_item, text=info)

        root.update()
        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # ESC
            break

    webcam.release()
    cv2.destroyAllWindows()
    root.destroy()

if __name__ == "__main__":
    main() 