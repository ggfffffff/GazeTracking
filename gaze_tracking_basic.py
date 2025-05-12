import cv2
import numpy as np
import json
import tkinter as tk
from gaze_tracking import GazeTracking

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
        print("校准文件未找到。请先运行 calibration.py 进行校准。")
        return None

def get_gaze_coordinates(hr, vr, calibration_data):
    """基础版本：简单的线性映射"""
    if not calibration_data:
        return None, None
    
    # 简单的线性映射
    x_vals = np.array([p["hr"] for p in calibration_data])
    y_vals = np.array([p["vr"] for p in calibration_data])
    screen_x_vals = np.array([p["x"] for p in calibration_data])
    screen_y_vals = np.array([p["y"] for p in calibration_data])
    
    # 线性拟合
    x_coef = np.polyfit(x_vals, screen_x_vals, 1)
    y_coef = np.polyfit(y_vals, screen_y_vals, 1)
    
    gaze_x = int(np.polyval(x_coef, hr))
    gaze_y = int(np.polyval(y_coef, vr))
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
    root.title("Gaze Tracking Basic")
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