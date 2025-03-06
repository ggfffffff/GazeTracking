import cv2
import numpy as np
import pyautogui
import json
from gaze_tracking import GazeTracking
from filterpy.kalman import KalmanFilter

# 校准文件路径
CALIBRATION_FILE = "calibration.json"

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
        return None, None  # 没有校准数据

    # 获取校准数据点
    x_vals = np.array([p["hr"] for p in calibration_data])
    y_vals = np.array([p["vr"] for p in calibration_data])
    screen_x_vals = np.array([p["x"] for p in calibration_data])
    screen_y_vals = np.array([p["y"] for p in calibration_data])

    # 线性拟合
    poly_x = np.polyfit(x_vals, screen_x_vals, 1)
    poly_y = np.polyfit(y_vals, screen_y_vals, 1)

    # 预测 gaze 坐标
    gaze_x = int(np.polyval(poly_x, hr))
    gaze_y = int(np.polyval(poly_y, vr))

    return gaze_x, gaze_y

def create_kalman_filter():
    """
    创建卡尔曼滤波器，用于平滑 gaze 追踪坐标
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 维状态（x, y, dx, dy），2 维观测（x, y）

    # 状态转移矩阵（假设匀速运动）
    kf.F = np.array([[1, 0, 1, 0], 
                     [0, 1, 0, 1], 
                     [0, 0, 1, 0], 
                     [0, 0, 0, 1]])

    # 观测矩阵（我们只能观测 x 和 y）
    kf.H = np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0]])

    # 过程噪声协方差矩阵（假设系统噪声较低）
    kf.P *= 1  # 初始不确定性
    kf.R *= 5  # 观测噪声
    kf.Q *= 0.01  # 过程噪声

    return kf

def main():
    """主程序：读取摄像头、计算 gaze 位置并显示"""
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    # 加载校准数据
    calibration_data = load_calibration()
    if not calibration_data:
        return  # 没有校准数据，退出程序

    # 配置 OpenCV 窗口，全屏显示
    cv2.namedWindow("Gaze Position", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaze Position", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    kf = create_kalman_filter()
    initialized = False  # 追踪初始化标志

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        # 处理当前帧
        gaze.refresh(frame)
        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()

        # 创建灰色背景
        display_frame = np.full((screen_height, screen_width, 3), 127, dtype=np.uint8)

        if hr is not None and vr is not None:
            gaze_x, gaze_y = get_gaze_coordinates(hr, vr, calibration_data)

            if gaze_x is not None and gaze_y is not None:
                if not initialized:
                    kf.x = np.array([gaze_x, gaze_y, 0, 0])  # 初始化滤波器状态
                    initialized = True

                # 卡尔曼预测 & 更新
                kf.predict()
                kf.update([gaze_x, gaze_y])

                # 获取平滑后的坐标
                filtered_x, filtered_y = int(kf.x[0]), int(kf.x[1])

                # 绘制 gaze 位置的红色点
                cv2.circle(display_frame, (filtered_x, filtered_y), 10, (0, 0, 255), -1)

                # 显示 gaze 位置数据
                text = f"Gaze: ({filtered_x}, {filtered_y})"
                cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        else:
            # 眼睛未检测到
            cv2.putText(display_frame, "Gaze: Not Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 显示窗口
        cv2.imshow("Gaze Position", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
            break

    webcam.release()
    cv2.destroyAllWindows()

# 运行主程序
if __name__ == "__main__":
    main()
