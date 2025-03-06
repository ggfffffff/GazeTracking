import cv2
import numpy as np
import pyautogui
import json
import time
from gaze_tracking import GazeTracking

CALIBRATION_FILE = "calibration.json"
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

calibration_points = [
    (screen_width // 2, screen_height // 2),
    (screen_width - 50, 50),
    (50, 50),
    (50, screen_height - 50),
    (screen_width - 50, screen_height - 50)
]

def calibrate():
    calibration_data = []
    min_samples = 10  # 每个点至少需要 10 组数据

    for cx, cy in calibration_points:
        while True:
            # 显示校准点 & 倒计时
            for remaining_time in range(5, 0, -1):
                display_frame = np.full((screen_height, screen_width, 3), 127, dtype=np.uint8)
                cv2.circle(display_frame, (cx, cy), 20, (0, 255, 0), -1)
                text = f"Look at the dot for {remaining_time} seconds"
                cv2.putText(display_frame, text, (cx - 200, cy - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("Calibration", display_frame)
                cv2.waitKey(1)
                time.sleep(1)

            # 采集 gaze 数据
            samples = []
            start_time = time.time()
            while time.time() - start_time < 5:
                ret, frame = webcam.read()
                if not ret:
                    continue

                gaze.refresh(frame)
                hr = gaze.horizontal_ratio()
                vr = gaze.vertical_ratio()

                if hr is not None and vr is not None:
                    samples.append((hr, vr))
                time.sleep(0.1)  # 采样间隔

            # 确保数据量足够
            if len(samples) < min_samples:
                print(f"⚠️ Not enough gaze data at ({cx}, {cy}). Retrying...")
                time.sleep(2)
                continue  # 重新采集该点

            # 计算平均 gaze 数据
            avg_hr = sum([s[0] for s in samples]) / len(samples)
            avg_vr = sum([s[1] for s in samples]) / len(samples)
            calibration_data.append({"hr": avg_hr, "vr": avg_vr, "x": cx, "y": cy})
            break  # 该点采集成功，跳出循环

    # 保存校准数据
    try:
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(calibration_data, f, indent=4)
        print(f"✅ Calibration complete! Data saved to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"❌ Error saving calibration file: {e}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate()
    webcam.release()
