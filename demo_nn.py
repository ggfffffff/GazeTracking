import cv2
import numpy as np
import pyautogui
import json
from gaze_tracking import GazeTracking
from filterpy.kalman import KalmanFilter
from tensorflow.keras.models import load_model
import gaze_nn  # 导入刚才创建的模块

# 获取屏幕分辨率
screen_width, screen_height = pyautogui.size()

def main():
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    
    # 加载校准数据（可以用作其他映射备用）
    calibration_data = gaze_nn.load_calibration()
    if not calibration_data:
        return
    
    # 尝试加载预训练的神经网络模型
    try:
        nn_model = load_model("gaze_nn_model.h5")
    except Exception as e:
        print("Pre-trained NN model not found. Please train the model by running gaze_nn.py.")
        return

    cv2.namedWindow("Gaze Position", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaze Position", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 这里保持原有 Kalman 滤波、低通滤波等逻辑不变，用 NN 模型代替线性回归映射
    def create_kalman_filter_acc():
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
        kf.R *= 20.
        kf.Q *= 0.01
        return kf

    kf = create_kalman_filter_acc()
    initialized = False
    last_filtered = None
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
            # 使用神经网络模型预测 gaze 坐标
            gaze_x, gaze_y = gaze_nn.get_gaze_coordinates_nn(hr, vr, nn_model)
            if gaze_x is not None and gaze_y is not None:
                if not initialized:
                    init_state = np.array([gaze_x, gaze_y, 0, 0, 0, 0])
                    kf.x = init_state
                    last_filtered = (gaze_x, gaze_y)
                    initialized = True

                kf.predict()
                kf.update([gaze_x, gaze_y])
                kalman_point = (int(kf.x[0]), int(kf.x[1]))

                # 此处可加入低通滤波和阈值判断，略同之前代码
                filtered_point = (kalman_point[0], kalman_point[1])
                if last_filtered is not None:
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
