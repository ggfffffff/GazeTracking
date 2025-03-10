import cv2
import numpy as np
import pyautogui
import json
import time
from gaze_tracking import GazeTracking

CALIBRATION_FILE = "calibration-12.json"
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 定义12个校准点：3行4列
rows = 3
cols = 4
margin_x = 50   # 左右边距
margin_y = 50   # 上下边距
calibration_points = []
for i in range(rows):
    for j in range(cols):
        x = int(margin_x + j * ((screen_width - 2 * margin_x) / (cols - 1)))
        y = int(margin_y + i * ((screen_height - 2 * margin_y) / (rows - 1)))
        calibration_points.append((x, y))

# 打印每个校准点的坐标
for idx, point in enumerate(calibration_points):
    print(f"Point {idx + 1}: {point}")

def calibrate():
    calibration_data = []
    min_samples = 10  # 每个点至少需要采集10组有效数据
    initial_radius = 10
    final_radius = 30  # 动画结束时的半径

    for cx, cy in calibration_points:
        print(f"Calibrating at point ({cx}, {cy})...")
        while True:
            samples = []
            active_time = 0.0  # 有效采样的累计时间（仅在检测到 gaze 时累加）
            prev_time = time.time()

            # 动画进行5秒（累计有效时间达到5秒）
            while active_time < 5.0:
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time

                ret, frame = webcam.read()
                if not ret:
                    continue

                # 更新 gaze 数据
                gaze.refresh(frame)
                hr = gaze.horizontal_ratio()
                vr = gaze.vertical_ratio()

                # 如果检测到 gaze，则累加有效时间并采集数据，否则暂停动画进程
                if hr is not None and vr is not None:
                    active_time += dt
                    samples.append((hr, vr))
                    gaze_detected = True
                else:
                    gaze_detected = False

                # 计算当前圆圈半径（基于有效时间线性插值）
                radius = int(initial_radius + (final_radius - initial_radius) * (active_time / 5.0))
                radius = min(radius, final_radius)

                # 剩余时间（向下取整）
                remaining_seconds = int(max(0, 5 - active_time))

                # 构建背景与动画
                display_frame = np.full((screen_height, screen_width, 3), 127, dtype=np.uint8)
                cv2.circle(display_frame, (cx, cy), radius, (0, 255, 0), -1)

                # 显示提示信息
                if gaze_detected:
                    text = f"Look at the dot for {remaining_seconds} seconds"
                else:
                    text = "Gaze not detected, pausing animation"
                cv2.putText(display_frame, text, (cx - 200, cy - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.imshow("Calibration", display_frame)
                cv2.waitKey(1)

            # 动画结束后检查采样数据量
            if len(samples) < min_samples:
                print(f"Not enough gaze data at ({cx}, {cy}). Retrying this point...")
                time.sleep(2)
                continue  # 重新采集该点
            else:
                # 计算平均 gaze 数据
                avg_hr = sum(s[0] for s in samples) / len(samples)
                avg_vr = sum(s[1] for s in samples) / len(samples)
                calibration_data.append({"hr": avg_hr, "vr": avg_vr, "x": cx, "y": cy})
                print(f"Calibration at ({cx}, {cy}) successful.")
                break  # 该点采集成功，退出重试循环

    # 保存校准数据
    try:
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(calibration_data, f, indent=4)
        print(f"Calibration complete! Data saved to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"Error saving calibration file: {e}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate()
    webcam.release()
