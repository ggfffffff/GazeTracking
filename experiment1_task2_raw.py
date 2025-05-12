import cv2
import numpy as np
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore
import os
from gaze_tracking import GazeTracking

class TrackingStabilityTaskRaw:
    def __init__(self, subject_name):
        self.screen_width = 1920
        self.screen_height = 1080
        self.target_radius = 10  # 目标点半径
        self.gaze_threshold = 100  # 注视判定阈值（像素）
        self.num_paths = 3  # 路径数量
        
        # 存储实验数据
        self.paths = []  # 路径数据
        self.gaze_points = []  # 注视点数据
        self.raw_gaze_points = []  # 原始注视点数据
        
        # 创建结果目录
        self.results_dir = f"E1T2_{subject_name}_raw"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # 眼动追踪相关参数
        self.calibration_file = "calibration-5.json"
        self.y_offset = -10
        
        # 初始化眼动追踪
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        self.calibration_data = self.load_calibration()
        
        # 生成路径
        self.generate_paths()
        
    def load_calibration(self):
        """加载校准数据"""
        try:
            with open(self.calibration_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("校准文件未找到，请先运行校准程序。")
            return None
            
    def get_gaze_coordinates(self, hr, vr):
        """根据校准数据，将gaze tracking的水平&垂直比例转换为屏幕坐标"""
        if not self.calibration_data:
            return None, None

        x_vals = np.array([p["hr"] for p in self.calibration_data])
        y_vals = np.array([p["vr"] for p in self.calibration_data])
        screen_x_vals = np.array([p["x"] for p in self.calibration_data])
        screen_y_vals = np.array([p["y"] for p in self.calibration_data])

        poly_x = np.polyfit(x_vals, screen_x_vals, 2)
        poly_y = np.polyfit(y_vals, screen_y_vals, 2)

        gaze_x = int(np.polyval(poly_x, hr))
        gaze_y = int(np.polyval(poly_y, vr)) + self.y_offset
        return gaze_x, gaze_y
        
    def generate_paths(self):
        """生成三条不同方向的路径"""
        margin = 200  # 边缘留白
        path_length = 800  # 路径长度
        
        # 水平路径
        start_x1 = margin
        start_y1 = self.screen_height // 2
        end_x1 = start_x1 + path_length
        end_y1 = start_y1
        self.paths.append({
            'start': (start_x1, start_y1),
            'end': (end_x1, end_y1),
            'duration': 4.0  # 4秒完成
        })
        
        # 45度斜向上路径
        start_x2 = margin
        start_y2 = self.screen_height - margin
        end_x2 = start_x2 + path_length
        end_y2 = start_y2 - path_length
        self.paths.append({
            'start': (start_x2, start_y2),
            'end': (end_x2, end_y2),
            'duration': 4.0
        })
        
        # 垂直路径
        start_x3 = self.screen_width // 2
        start_y3 = self.screen_height - margin
        end_x3 = start_x3
        end_y3 = start_y3 - path_length
        self.paths.append({
            'start': (start_x3, start_y3),
            'end': (end_x3, end_y3),
            'duration': 4.0
        })
        
    def calculate_tde(self, target_points, gaze_points):
        """计算轨迹偏移误差（Trajectory Deviation Error）"""
        errors = []
        for target, gaze in zip(target_points, gaze_points):
            error = np.sqrt((target[0] - gaze[0])**2 + (target[1] - gaze[1])**2)
            errors.append(error)
        return np.mean(errors)
        
    def calculate_smoothness(self, points):
        """计算平滑度（相邻点之间速度变化程度）"""
        if len(points) < 3:
            return 0
            
        velocities = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            velocity = np.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
            
        # 计算速度变化的标准差
        velocity_changes = np.diff(velocities)
        return np.std(velocity_changes)
        
    def run_experiment(self):
        """运行实验"""
        if not self.calibration_data:
            print("未找到校准数据，请先运行校准程序。")
            return
            
        # 创建全屏窗口
        cv2.namedWindow("Experiment", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Experiment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        for path_idx, path in enumerate(self.paths):
            print(f"开始第 {path_idx + 1} 条路径...")
            
            # 计算路径点
            start_point = path['start']
            end_point = path['end']
            duration = path['duration']
            
            # 计算移动速度
            dx = (end_point[0] - start_point[0]) / (duration * 30)  # 假设30fps
            dy = (end_point[1] - start_point[1]) / (duration * 30)
            
            # 当前路径的数据
            current_path_points = []
            current_gaze_points = []
            current_raw_gaze_points = []
            
            # 开始移动
            current_x, current_y = start_point
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration:
                # 更新目标点位置
                current_x += dx
                current_y += dy
                current_point = (int(current_x), int(current_y))
                current_path_points.append(current_point)
                
                # 获取原始注视点
                ret, frame = self.webcam.read()
                if not ret:
                    continue
                    
                self.gaze.refresh(frame)
                hr = self.gaze.horizontal_ratio()
                vr = self.gaze.vertical_ratio()
                
                if hr is not None and vr is not None:
                    gaze_x, gaze_y = self.get_gaze_coordinates(hr, vr)
                    if gaze_x is not None and gaze_y is not None:
                        current_gaze_points.append((gaze_x, gaze_y))
                        current_raw_gaze_points.append((gaze_x, gaze_y))
                
                # 显示画面
                frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                cv2.circle(frame, current_point, self.target_radius, (0, 255, 0), -1)
                cv2.imshow("Experiment", frame)
                
                # 控制帧率
                frame_count += 1
                time.sleep(max(0, 1/30 - (time.time() - start_time - frame_count/30)))
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
                    return
                    
            # 保存当前路径的数据
            self.paths[path_idx]['points'] = current_path_points
            self.gaze_points.append(current_gaze_points)
            self.raw_gaze_points.append(current_raw_gaze_points)
            
            # 路径间暂停
            time.sleep(1)
            
        cv2.destroyAllWindows()
        self.webcam.release()
        
    def calculate_metrics(self):
        """计算评价指标"""
        metrics = []
        for path_idx, path in enumerate(self.paths):
            target_points = path['points']
            gaze_points = self.gaze_points[path_idx]
            
            # 计算TDE
            tde = self.calculate_tde(target_points, gaze_points)
            
            # 计算平滑度
            smoothness = self.calculate_smoothness(gaze_points)
            
            metrics.append({
                'path_index': path_idx + 1,
                'TDE': tde,
                'smoothness': smoothness
            })
            
        return metrics
        
    def generate_visualization(self):
        """生成轨迹可视化图"""
        plt.figure(figsize=(15, 5))
        
        for path_idx, path in enumerate(self.paths):
            plt.subplot(1, 3, path_idx + 1)
            
            # 绘制理想轨迹
            target_points = path['points']
            target_x = [p[0] for p in target_points]
            target_y = [p[1] for p in target_points]
            plt.plot(target_x, target_y, 'g-', label='理想轨迹', alpha=0.5)
            
            # 绘制原始注视轨迹
            gaze_points = self.gaze_points[path_idx]
            gaze_x = [p[0] for p in gaze_points]
            gaze_y = [p[1] for p in gaze_points]
            plt.plot(gaze_x, gaze_y, 'r.', label='原始注视点', alpha=0.3)
            
            plt.title(f'路径 {path_idx + 1}')
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'trajectory_comparison.png'))
        plt.close()
        
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "timestamp": timestamp,
            "metrics": self.calculate_metrics(),
            "paths": self.paths,
            "gaze_points": self.gaze_points,
            "raw_gaze_points": self.raw_gaze_points
        }
        
        # 保存JSON格式的完整数据
        with open(os.path.join(self.results_dir, f'results_{timestamp}.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # 保存CSV格式的原始数据
        for path_idx, path in enumerate(self.paths):
            raw_data = np.column_stack((
                [p[0] for p in path['points']],
                [p[1] for p in path['points']],
                [p[0] for p in self.gaze_points[path_idx]],
                [p[1] for p in self.gaze_points[path_idx]]
            ))
            np.savetxt(
                os.path.join(self.results_dir, f'raw_data_path{path_idx+1}_{timestamp}.csv'),
                raw_data,
                delimiter=',',
                header='target_x,target_y,gaze_x,gaze_y',
                comments=''
            )
            
    def run(self):
        """运行完整的实验流程"""
        print("开始实验...")
        self.run_experiment()
        print("计算评价指标...")
        metrics = self.calculate_metrics()
        for m in metrics:
            print(f"路径 {m['path_index']}:")
            print(f"  TDE: {m['TDE']:.2f}像素")
            print(f"  平滑度: {m['smoothness']:.2f}")
        print("生成可视化...")
        self.generate_visualization()
        print("保存结果...")
        self.save_results()
        print("实验完成！")

if __name__ == "__main__":
    # 在这里修改受试者姓名
    subject_name = "test_subject"
    experiment = TrackingStabilityTaskRaw(subject_name)
    experiment.run() 