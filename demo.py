import cv2
import numpy as np
import pyautogui
import json
import time
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

def create_kalman_filter_acc():
    """
    创建一个基于常加速模型的卡尔曼滤波器，状态向量为 [x, y, vx, vy, ax, ay]
    """
    kf = KalmanFilter(dim_x=6, dim_z=2)
    dt = 1.0
    kf.F = np.array([[1, 0, dt, 0, 0.5*dt*dt, 0],
                     [0, 1, 0, dt, 0, 0.5*dt*dt],
                     [0, 0, 1, 0, dt, 0],
                     [0, 0, 0, 1, 0, dt],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0]])
    kf.P *= 10.
    kf.R *= 20.   # 增大测量噪声，使滤波器平滑效果更强
    kf.Q *= 0.01
    return kf

def low_pass_filter(new_point, last_filtered, alpha_x=0.2, alpha_y=0.1):
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

def main():
    """主程序：读取摄像头、计算 gaze 位置并显示最终平滑后的结果"""
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    calibration_data = load_calibration()
    if not calibration_data:
        return

    cv2.namedWindow("Gaze Position", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaze Position", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    kf = create_kalman_filter_acc()
    initialized = False

    # 用于记录上一次低通滤波后的结果
    last_filtered = None
    # 阈值：如果连续帧变化小于这个阈值，则保持上一帧结果
    delta_threshold = 5

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        gaze.refresh(frame)
        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()

        display_frame = np.full((screen_height, screen_width, 3), 127, dtype=np.uint8)

        if hr is not None and vr is not None:
            gaze_x, gaze_y = get_gaze_coordinates(hr, vr, calibration_data)
            if gaze_x is not None and gaze_y is not None:
                if not initialized:
                    init_state = np.array([gaze_x, gaze_y, 0, 0, 0, 0])
                    kf.x = init_state
                    last_filtered = (gaze_x, gaze_y)
                    initialized = True

                kf.predict()
                kf.update([gaze_x, gaze_y])
                kalman_point = (int(kf.x[0]), int(kf.x[1]))

                # 对卡尔曼结果再做低通滤波，y方向使用更小的alpha实现更强平滑
                filtered_point = low_pass_filter(kalman_point, last_filtered, alpha_x=0.2, alpha_y=0.1)

                # 阈值判断：如果变化非常小，则保持上一帧的结果
                dx = abs(filtered_point[0] - last_filtered[0])
                dy = abs(filtered_point[1] - last_filtered[1])
                if dx < delta_threshold and dy < delta_threshold:
                    filtered_point = last_filtered

                last_filtered = filtered_point

                cv2.circle(display_frame, filtered_point, 10, (0, 0, 255), -1)
                text = f"Gaze: {filtered_point}"
                cv2.putText(display_frame, text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Gaze: Not Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Gaze Position", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
