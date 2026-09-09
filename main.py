# In[]

#main
import torch
from utils.torch_utils import select_device
import time
import pandas as pd

from realsense_camera import Camera
from yolo_detector import ObjectDetect
from inverse_kinematics import InverseKinematicsSolver
from dorna_controller import RobotController
from llm_api import LLM
from label_driven_motion import Processor
from vision_coordinator import Coordinator

if __name__ == "__main__":
    try:
        output_dir = r""
        device = select_device('')
        camera = Camera()
        robot_ip = "ROBOT_IP"
        bot = RobotController(robot_ip)
        ik_solver = InverseKinematicsSolver()
        yolo = ObjectDetect(
            device=device,
            model_path=r"yolov7.pt",
            output_dir=output_dir,
            ik_solver=ik_solver
        )

        model = yolo.get_model()
        coco_labels = model.names
        llm = LLM(coco_labels)
        processor = Processor(llm, ik_solver, bot)
        #processor = Processor(llm, ik_solver)
        processor.start()

        '''
        bot.set_joint(j0=180, j1=135, j2=-135)
        bot.set_joint(j4=363)
        bot.set_joint(j3=120, j4=360)
        bot.set_joint(j3=85)
        bot.set_joint(j0=175)
        '''

        # 控制器來協調
        coordinator = Coordinator(camera, yolo, processor)

        while True:
            if camera.stop_event.is_set():
                print("程式結束")
                break
            bot.move_to_home()
            processor.start()
            label = input("請輸入指令: ")
            llm.set_label(label)
            result = llm.run()
            processor.set_target_labels(result)
            coordinator.start()  # 進入處理流程
            time.sleep(2)
            #bot.get_current_position()

    except KeyboardInterrupt:
        print("程式被中斷")
    except Exception as e:
        print(f"系統錯誤: {e}")
    finally:
        #if 'bot' in locals():
            #bot.disconnect()
        if 'camera' in locals():
            camera.stop()

