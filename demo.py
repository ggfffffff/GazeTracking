import cv2
import numpy as np
import pyautogui
from gaze_tracking import GazeTracking
from filterpy.kalman import KalmanFilter

# 初始化 GazeTracking
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# 获取屏幕分辨率
screen_width, screen_height = pyautogui.size()

# 配置窗口，全屏显示
cv2.namedWindow("Gaze Position", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Gaze Position", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 初始化卡尔曼滤波器
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4维状态（x, y, dx, dy），2维观测（x, y）
    
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

kf = create_kalman_filter()

# 追踪初始化
initialized = False

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # 分析当前帧
    gaze.refresh(frame)
    hr = gaze.horizontal_ratio()  # 水平比例：[0,1]，0=极右, 1=极左
    vr = gaze.vertical_ratio()    # 垂直比例：[0,1]，0=极上, 1=极下

    # 建一个灰色背景
    display_frame = np.full((screen_height, screen_width, 3), 127, dtype=np.uint8)

    # 计算卡尔曼滤波
    if hr is not None and vr is not None:
        # 计算屏幕坐标
        gaze_x = int((1 - hr) * screen_width)
        gaze_y = int(vr * screen_height)

        if not initialized:
            # 初始化卡尔曼滤波器状态
            kf.x = np.array([gaze_x, gaze_y, 0, 0])  # 初始状态 (x, y, dx, dy)
            initialized = True

        # 卡尔曼预测
        kf.predict()

        # 观测更新
        kf.update([gaze_x, gaze_y])

        # 获取平滑坐标
        filtered_x, filtered_y = int(kf.x[0]), int(kf.x[1])

        # 绘制红色点
        cv2.circle(display_frame, (filtered_x, filtered_y), 10, (0, 0, 255), -1)

        # 显示数值，便于调试
        text = f"Gaze: ({filtered_x}, {filtered_y})"
        cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        # 未检测到，提示
        cv2.putText(display_frame, "Gaze: Not Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # 显示窗口
    cv2.imshow("Gaze Position", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
