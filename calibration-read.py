# calibration-read.py
import cv2
import numpy as np
import pyautogui
import json
import time
from gaze_tracking import GazeTracking
from PIL import ImageFont, ImageDraw, Image  # 导入 Pillow 库

# 校准文件保存路径
CALIBRATION_FILE = "calibration-read.json"
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# 设置全屏窗口
cv2.namedWindow("Reading Calibration", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Reading Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 定义要显示的文本（10行示例文本）
text_lines = [
    "欢迎使用眼动追踪系统。",
    "请仔细阅读下面的说明。",
    "本系统通过检测您的眼动来估计视线位置。",
    "当您阅读这段文字时，系统会自动校准。",
    "请保持视线自然移动，沿着文本的行列阅读。",
    "您无需刻意注视每个字。",
    "系统将根据您的自然阅读行为建立映射。",
    "校准完成后，您可以使用眼动控制程序。",
    "确保光线充足并保持正对摄像头。",
    "阅读结束后，系统会自动保存校准数据。"
]

# 文本显示设置
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2
line_spacing = 50   # 行间距（像素）
# 文本区域起始 y 坐标（使文本整体垂直居中）
total_text_height = len(text_lines) * line_spacing
start_y = (screen_height - total_text_height) // 2

# 预先计算每一行文本的开头和结尾坐标（用于校准映射）
calib_text_positions = []
for i, line in enumerate(text_lines):
    # 计算文本大小
    (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
    # 计算 x 坐标，使文本水平居中
    x = (screen_width - w) // 2
    # y 坐标
    y = start_y + i * line_spacing + h // 2
    # 记录该行的开始和结束坐标
    start_x = x
    end_x = x + w
    calib_text_positions.append(((start_x, y), (end_x, y)))

# 在屏幕上显示文本（保持静态背景，便于阅读）
background = np.full((screen_height, screen_width, 3), 255, dtype=np.uint8)

# 使用 Pillow 绘制中文文本
def put_chinese_text(img, text, position, font_size=30, color=(0, 0, 0)):
    # 使用 Pillow 创建一个 Image 对象
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # 使用 SimHei 字体（可以更换为其他支持的中文字体路径）
    font = ImageFont.truetype("msyh.ttc", font_size)  # 使用 msyh.ttc，Windows 自带的字体
    draw.text(position, text, font=font, fill=color)
    
    # 将 PIL 图像转换回 OpenCV 图像
    img = np.array(pil_img)
    return img

for i, line in enumerate(text_lines):
    # 绘制文本
    y = start_y + i * line_spacing
    background = put_chinese_text(background, line, (screen_width // 2 - len(line) * 10, y), font_size=40, color=(0, 0, 0))

# 显示文本界面，等待几秒钟让用户适应
cv2.imshow("Reading Calibration", background)
cv2.waitKey(1000)

# 开始记录 gaze 数据（记录30秒）
record_duration = 30.0  # 秒
recorded_data = []  # 记录元组：(timestamp, hr, vr)
start_time = time.time()
print("开始记录眼动数据，请开始阅读...")
while time.time() - start_time < record_duration:
    ret, frame = webcam.read()
    if not ret:
        continue
    gaze.refresh(frame)
    hr = gaze.horizontal_ratio()
    vr = gaze.vertical_ratio()
    current_time = time.time()
    if hr is not None and vr is not None:
        recorded_data.append((current_time, hr, vr))
    # 为了保证界面响应
    cv2.imshow("Reading Calibration", background)
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
print("记录结束，正在处理数据...")

# 将记录的数据按时间均匀分成与文本行数相同的段数（此处为10段）
num_lines = len(text_lines)
if len(recorded_data) == 0:
    print("未检测到有效眼动数据，请重试。")
    exit()

# 取起始和结束时间
t0 = recorded_data[0][0]
t_end = recorded_data[-1][0]
segment_duration = (t_end - t0) / num_lines

calibration_data = []
for i in range(num_lines):
    seg_start = t0 + i * segment_duration
    seg_end = seg_start + segment_duration
    seg_samples = [(hr, vr) for (t, hr, vr) in recorded_data if seg_start <= t < seg_end]
    if len(seg_samples) == 0:
        print(f"第 {i+1} 段无数据，跳过。")
        continue
    avg_hr = sum(s[0] for s in seg_samples) / len(seg_samples)
    avg_vr = sum(s[1] for s in seg_samples) / len(seg_samples)
    # 取该行的开始和结束坐标作为期望值
    expected_coord_start = calib_text_positions[i][0]
    expected_coord_end = calib_text_positions[i][1]
    calibration_data.append({
        "hr": avg_hr,
        "vr": avg_vr,
        "x_start": expected_coord_start[0],
        "y_start": expected_coord_start[1],
        "x_end": expected_coord_end[0],
        "y_end": expected_coord_end[1]
    })
    print(f"校准数据第 {i+1} 点: hr={avg_hr:.3f}, vr={avg_vr:.3f} -> ({expected_coord_start[0]}, {expected_coord_start[1]}) 到 ({expected_coord_end[0]}, {expected_coord_end[1]})")

# 保存校准数据到 JSON 文件
try:
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calibration_data, f, indent=4)
    print(f"隐式校准完成，数据已保存到 {CALIBRATION_FILE}")
except Exception as e:
    print(f"保存校准数据出错: {e}")
