import cv2
import numpy as np
import json
import tkinter as tk
from gaze_tracking import GazeTracking
from filterpy.kalman import KalmanFilter

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

def main():
    """主程序：读取摄像头、计算 gaze 位置并在透明窗口上显示"""
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

    # ----------- 创建透明 Tkinter 窗口 -----------
    root = tk.Tk()
    root.title("Gaze Tracking Kalman")
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
                # 初始化卡尔曼滤波状态
                if not initialized:
                    kf.x = np.array([gaze_x, gaze_y, 0, 0, 0, 0])
                    initialized = True

                # 卡尔曼预测 & 更新
                kf.predict()
                kf.update([gaze_x, gaze_y])
                kalman_point = (int(kf.x[0]), int(kf.x[1]))

                canvas.coords(dot,
                            kalman_point[0] - dot_radius,
                            kalman_point[1] - dot_radius,
                            kalman_point[0] + dot_radius,
                            kalman_point[1] + dot_radius)
                info = f"Gaze: ({kalman_point[0]}, {kalman_point[1]})"
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