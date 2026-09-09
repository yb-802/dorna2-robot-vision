import torch 
from utils.torch_utils import select_device
import time
import threading
import queue


class Processor:
    '''
    llm、計算座標、robot整合
    '''
    def __init__(self, llm, ik_solver, bot):
    #def __init__(self, llm, ik_solver):
        self.llm = llm
        self.ik_solver = ik_solver
        self.bot = bot
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.target_labels = []
        self.processing_done = threading.Event()
        self.processing_done.set()

    def start(self):
        threading.Thread(target=self.process_loop, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        
    def set_target_labels(self, labels):
        """由主程式設定目標標籤"""
        self.target_labels = labels

    def add_result(self, result):
        # 把result放入queue
        self.result_queue.put(result)

    def process_loop(self):
        while not self.stop_event.is_set():
            if not self.processing_done.is_set():
                time.sleep(0.1)
                continue

            try:
                result = self.result_queue.get(timeout=1)
                self.processing_done.clear()
                threading.Thread(target=self.handle_result, args=(result,), daemon=True).start()
            except queue.Empty:
                continue

    def handle_result(self, result):
        try:
            if not self.target_labels:
                print("[Processor] 尚未設定目標標籤")
                return
            print(f"[LLM] Label: {self.target_labels}")
            
            # 過濾 YOLO 結果中 label 有出現在使用者輸入的目標物件
            matched_objects = [obj for obj in result if obj["label"] in self.target_labels]
            
            if not matched_objects:
                print("[LLM] 沒有在 YOLO 結果中找到相符的物件。")
                return
            
            # 依序處理每個目標物件
            for obj in matched_objects:
                x, y, z = obj["coord"]
                x = x + 110
                y = y + 20
                z = z + 240
                a = 50  # 預設角度
                print(f"[Target] 處理 {obj['label']} at ({x:.1f}, {y:.1f}, {z:.1f})")
    
                j0, j1, j2, j3 = self.ik_solver.compute_angle_3dof(x, y, z, a)
                print(f"j0={j0:.3f}, j1={j1:.3f}, j2={j2:3f}, j3={j3:.3f}")
                self.bot.set_joint(j3=j3)
                self.bot.set_joint(j0=j0, j1=j1, j2=j2)
                #time.sleep(5)
                #self.bot.move_to_home()

            time.sleep(3)  # 假設執行時間
            self.bot.get_current_position()
        except Exception as e:
            print(f"[Processor Error] {e}")
        finally:
            self.processing_done.set()

