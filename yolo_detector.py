# In[]

# yolo_detector
import numpy as np
import cv2
import os
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox
import pandas as pd
import pyrealsense2 as rs
import traceback
import gc
import math


class ObjectDetect:
    """
    YOLO辨識
    """
    def __init__(self, device, model_path, output_dir, ik_solver):
        if isinstance(device, str):
            self.device = select_device(device)
        else:
            self.device = device
            
        self.model = attempt_load(model_path, map_location=self.device)
        self.model.eval()
        # 優化：設定半精度推論（如果支援）
        if self.device.type != 'cpu':
            self.model.half()  # 使用 FP16
            self.use_half = True
        else:
            self.use_half = False
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.output_dir = output_dir
        self._setup_output_dirs()   
        self._warmup_model()
        self.inverse_kinematics = ik_solver

    def _setup_output_dirs(self):
        """
        預先建立輸出目錄
        """
        dirs = [
            os.path.join(self.output_dir, "d435_output"),
            os.path.join(self.output_dir, "d435_output", "camera_file"),
            os.path.join(self.output_dir, "d435_output", "detect_file")
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _warmup_model(self):
        """
        模型暖身，提升首次推論速度
        """
        print("模型暖身中...")
        dummy_input = torch.zeros(1, 3, 416, 416).to(self.device)  # 使用較小的輸入尺寸
        if self.use_half:
            dummy_input = dummy_input.half()
        
        with torch.no_grad():
            for _ in range(3):  # 執行幾次暖身
                _ = self.model(dummy_input)
        
        # 清理記憶體
        del dummy_input
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("模型暖身完成")

    def capture_image(self, depth, color, intrinsics):
        """
        優化版影像處理與辨識
        """
        print("開始 YOLO 檢測...")
        
        # 減少不必要的複製
        color_copy = color.copy()
        
        # 儲存原始檔案
        self._save_raw_files(depth, color)

        img, processed_color = self.process_image(color_copy)
        det = self.detect(img, processed_color, color.shape)  # 傳入原始形狀
        results = self.load_results(depth, det, intrinsics)
        
        # 優化：記憶體清理
        del img, processed_color
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return results

    def _save_raw_files(self, depth, color):
        """
        儲存原始檔案
        """
        out_dir = os.path.join(self.output_dir, "d435_output", "camera_file")
        np.save(os.path.join(out_dir, "depth.npy"), depth)
        cv2.imwrite(os.path.join(out_dir, "color.jpg"), color)

    def process_image(self, color_image):
        """
        優化版影像前處理
        """
        # 使用較小的輸入尺寸提升速度
        img = letterbox(color_image, new_shape=416)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        
        # 根據設備選擇精度
        if self.use_half:
            img = img.half() / 255.0
        else:
            img = img.float() / 255.0
            
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, color_image

    def detect(self, img, color_image, original_shape):
        """
        優化版辨識
        """
        with torch.no_grad():
            # 降低推論參數以提升速度
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, 0.4, 0.5)  # 提高閾值過濾更多框
        
        return self.process_detect(img, color_image, pred, original_shape)

    def process_detect(self, img, color_image, pred, original_shape):
        """
        優化版辨識結果處理
        """
        results = []
        for det in pred:
            if len(det):
                # 優化：座標縮放到原始影像尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_shape).round()
                
                for *xyxy, conf, cls in det:
                    label = f'{self.names[int(cls)]}'
                    x1, y1, x2, y2 = map(int, xyxy)
                    results.append({
                        'label': label, 'confidence': float(conf),
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    })
                    
                    # 優化：簡化繪圖
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 較小字體
        
        # 儲存結果
        self._save_results(color_image, results)
        return results

    def _save_results(self, color_image, results):
        """
        儲存辨識結果
        """
        out_dir = os.path.join(self.output_dir, "d435_output", "detect_file")
        cv2.imwrite(os.path.join(out_dir, "yolov7.jpg"), color_image)
        if results:  # 只有在有結果時才儲存 CSV
            pd.DataFrame(results).to_csv(os.path.join(out_dir, "det.csv"), 
                                       encoding="utf-8-sig", index=False)

    def load_results(self, depth_array, det, intrinsics):
        """
        優化版結果輸出
        """
        try:
            if not det:
                print("未偵測到任何物件")
                return
                
            det_df = pd.DataFrame(det)
            print(f"\n=== 偵測結果 ({len(det_df)} 個物件) ===")
            
            results = []
            for idx, row in det_df.iterrows():
                # 優化：邊界檢查
                x1, y1, x2, y2 = self._safe_bounds(
                    int(row['x1']), int(row['y1']), 
                    int(row['x2']), int(row['y2']), 
                    depth_array.shape
                )
                
                # 計算中心點和實際座標
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                center_depth = depth_array[center_y, center_x]
                
                if center_depth > 0:  # 確保深度值有效
                    actual_x, actual_y, actual_z = self.actual_coordinates(
                        intrinsics, center_x, center_y, center_depth / 1000.0)
                    
                    result = {
                        "label": row["label"],
                        "coord": (actual_x, actual_y, actual_z)
                    }
                
                    results.append(result)
                    
                    print(f"物件 {idx+1}: {row['label']} (信心度: {row['confidence']:.2f})")
                    print(f"  中心座標: ({center_x}, {center_y})")
                    print(f"  中心深度: {center_depth}mm")
                    print(f"  實際座標: ({actual_x:.3f}, {actual_y:.3f}, {actual_z:.3f})")
                    print("-" * 30)
                else:
                    print(f"物件 {idx+1}: {row['label']} - 深度值無效")
            
            out_dir = os.path.join(self.output_dir, "d435_output", "camera_file")
            pd.DataFrame(results).to_csv(os.path.join(out_dir, "results.csv"), 
                                       encoding="utf-8-sig", index=False)
            return results
                  
        except Exception as e:
            print(f"載入檔案時發生錯誤: {e}")
            traceback.print_exc()

    def _safe_bounds(self, x1, y1, x2, y2, shape):
        """
        安全的邊界檢查
        """
        h, w = shape
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        return x1, y1, x2, y2

    def actual_coordinates(self, intrinsics, x, y, z):
        """
        將影像像素轉換為世界座標（考慮旋轉 + 單純平移）
        """
        # 轉為相機座標系下的點 (m)
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], z)
        Xc, Yc, Zc = [p * 100 for p in point_3d]  # 轉為 cm
    
        # === 根據相機安裝的方式設置這個旋轉矩陣 ===
        theta = math.radians(-60)   #相機向下60度
        Rotate = np.array([[1, 0, 0],
                           [0,  math.cos(theta), -math.sin(theta)],
                           [0,  math.sin(theta),  math.cos(theta)]])
        R = np.array([
            [0, 0, 1],     # z_cam → x_world
            [-1, 0, 0],    # x_cam → y_world
            [0, -1, 0],    # y_cam → z_world
        ])
    
        P_cam = np.array([[Xc], [Yc], [Zc]])
        P_cam = np.dot(Rotate, P_cam)
        P_world = np.dot(R, P_cam)
    
        xw, yw, zw = P_world.flatten()
        return xw*10, yw*10, zw*10

    def get_model(self):
        return self.model
    
    
    
