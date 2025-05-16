# gaze_nn.py
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

CALIBRATION_FILE = "calibration.json"

def load_calibration():
    """
    加载校准数据文件，返回校准数据列表。
    校准数据格式示例：
      [
          {"hr": 0.52, "vr": 0.48, "x": 960, "y": 540},
          ...
      ]
    """
    try:
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Calibration file not found. Please run calibration.py first.")
        return None

def build_nn_model(input_dim=2, hidden_units=16, hidden_layers=2):
    """
    构建一个简单的前馈神经网络模型，用于将 (hr, vr) 映射到屏幕坐标 (x, y)。
    参数：
      input_dim: 输入特征维度，默认为2 (hr, vr)
      hidden_units: 每个隐藏层神经元数量
      hidden_layers: 隐藏层数量
    返回训练编译好的模型。
    """
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_dim=input_dim))
    for _ in range(hidden_layers - 1):
        model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(2))  # 输出层：预测 x, y 坐标（回归任务）
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_nn_model(calibration_data, epochs=500, hidden_units=16, hidden_layers=2):
    """
    用校准数据训练神经网络映射模型。
    参数：
      calibration_data: 校准数据列表，每个元素包含 "hr", "vr", "x", "y"
      epochs: 训练轮数，默认500
      hidden_units: 每个隐藏层的神经元数量
      hidden_layers: 隐藏层数量
    返回训练好的模型。
    """
    X = np.array([[p["hr"], p["vr"]] for p in calibration_data])
    Y = np.array([[p["x"], p["y"]] for p in calibration_data])
    model = build_nn_model(input_dim=2, hidden_units=hidden_units, hidden_layers=hidden_layers)
    model.fit(X, Y, epochs=epochs, verbose=0)
    return model

def get_gaze_coordinates_nn(hr, vr, model):
    """
    利用训练好的神经网络模型预测 gaze 坐标。
    参数：
      hr, vr: 当前 gaze tracking 返回的水平和垂直比例（标量）
      model: 训练好的神经网络模型
    返回预测的 (x, y) 坐标（整数）。
    """
    input_data = np.array([[hr, vr]])
    pred = model.predict(input_data, verbose=0)
    gaze_x, gaze_y = int(pred[0][0]), int(pred[0][1])
    
    # 添加边界限制
    gaze_x = max(0, min(gaze_x, screen_width))
    gaze_y = max(0, min(gaze_y, screen_height))
    
    return gaze_x, gaze_y

if __name__ == "__main__":
    # 模块直接运行时，自动加载校准数据并训练模型，然后保存模型
    calibration_data = load_calibration()
    if calibration_data is None or len(calibration_data) < 5:
        print("Insufficient calibration data! Please perform calibration.")
    else:
        model = train_nn_model(calibration_data, epochs=500)
        model.save("gaze_nn_model.h5")
        print("Neural network model trained and saved as gaze_nn_model.h5.")
        # 测试：使用校准数据中的第一个点进行预测
        test_hr = calibration_data[0]["hr"]
        test_vr = calibration_data[0]["vr"]
        pred_x, pred_y = get_gaze_coordinates_nn(test_hr, test_vr, model)
        print(f"Test prediction for first calibration point: ({pred_x}, {pred_y})")
