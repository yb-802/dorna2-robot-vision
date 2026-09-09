# realsense_camera
import numpy as np
import cv2
import pyrealsense2 as rs
import threading
import queue
import traceback


class Camera:
    def __init__(self):
        self.pipeline = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.detecting_lock = threading.Lock()

    def init_camera(self):
        """
        初始化 RealSense 相機參數與管線設定。
        """
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # 長:640、寬:480、每秒影格數數:15
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.pipeline.start(config)

    def get_frame(self):
        """
        取得對齊後的深度與彩色影像
        """
        try:
            while not self.stop_event.is_set():
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError as e:
                    print(f"等待影格超時，繼續下一輪: {e}")
                    continue  # 直接跳過，繼續等待下一幀
                
                frames = self.pipeline.wait_for_frames()
                aligned = rs.align(rs.stream.color).process(frames)
                depth_frame = aligned.get_depth_frame() # 對齊到彩色影像
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                if self.frame_queue.full(): # queue滿了，丟掉之前影像
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put((depth_frame, color_frame))    # 將深度、彩色圖放入queue
        except Exception as e:
            print("Get frame error:", e)
            traceback.print_exc()

    def display_frame(self):
        """
        顯示影像
        :return: (深度影像(NumPy), 彩色影像(NumPy), 相機內部參數) 或 (None, None, None)
        """
        if not self.frame_queue.empty():
            depth_frame, color_frame = self.frame_queue.queue[-1]   # queue的最新影像
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()   # 相機內部參數
            cv2.imshow("Color", color)
            return depth, color, intrinsics
        return None, None, None

    def stop(self):
        """
        停止相機串流。
        """
        self.stop_event.set()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"Stopping pipeline error: {e}")
            self.pipeline = None  # 空 pipeline，確保下次會重新 init
            
    def run(self, detect_callback):
        self.init_camera()
        frame_thread = threading.Thread(target=self.get_frame, daemon=True)
        frame_thread.start()
    
        while not self.stop_event.is_set():
            depth, color, intrinsics = self.display_frame()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a') and self.detecting_lock.acquire(blocking=False): # 按 'q' 擷取影像開始辨識
                if color is not None and depth is not None:
                    detect_thread = threading.Thread(
                        target=self._detect_worker, 
                        args=(detect_callback, depth.copy(), color.copy(), intrinsics),
                        daemon=True
                    )
                    detect_thread.start()
                else:
                    self.detecting_lock.release()
            elif key == ord('q'):  # 按 ESC 結束程式
                cv2.destroyAllWindows()
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                self.stop()
                # 這裡設定 stop_event，讓外層迴圈能判斷是否要退出
                self.stop_event.set()
                break
    
    def _detect_worker(self, detect_callback, depth, color, intrinsics):
        """
        在背景執行緒中執行檢測
        """
        try:
            detect_callback(depth, color, intrinsics)
        except Exception as e:
            print(f"Detection error: {e}")
            traceback.print_exc()
        finally:
            self.detecting_lock.release()