# dorna_controller
import time
from dorna2 import Dorna
import math


class RobotController:
    """
    控制 Dorna2 機械手臂的封裝類別，提供初始化、錯誤處理、與移動控制功能。
    """

    def __init__(self, robot_ip):
        """
        初始化連線與初始設定。
        """
        self.robot = Dorna()
        self.robot.connect(robot_ip)
        # self.robot.register_callback(self.alarm_condition)  # 註冊 alarm callback 等待測試
        self._reset_robot()

        # 初始角度位置（單位：角度）
        self.initial_joint_pos = {
            "j0": 180,    # +：右轉
            "j1": 180,    # -：往上抬
            "j2": -142,   # +：往上壓
            "j3": 135,    # +：夾子往上台
            "j4": 315       # +：夾子向右彎 (會因為看的角度不同有差異自行判斷)
        }
        
        self.home_pos = {
            "j0": 175,    
            "j1": 135,    
            "j2": -135,   
            "j3": 85,    # J3和J4有時候會跑掉，要確認
            "j4": 363
        }

        # 啟始夾爪位置調整
        #self.set_joint(j3=180)

    def _reset_robot(self):
        """
        嘗試解除 alarm 並啟動 motor。
        """
        try:
            self.robot.set_alarm(0)
            time.sleep(0.2)
            result = self.robot.set_motor(1)
            if result < 0:
                self._handle_error("Motor enable failed")
        except Exception as e:
            print(f"初始化錯誤：{e}")
            self.robot.log("activating alarm")

    def alarm_condition(self, event, value):
        """
        機器人發生 alarm 時的處理函數。
        """
        if event != "alarm":
            return  # 忽略非 alarm 的事件

        # 若 value 重複，就不再處理
        print(f"⚠️ Alarm triggered: {value}")
        
        if isinstance(value, dict) and "code" in value:
            code = value["code"]
            if code == 3:
                print("👉 嘗試重新啟動 motor")
                self._reset_robot()
            else:
                print(f"🚨 未知錯誤碼: {code}")


    def _handle_error(self, message="未知錯誤"):
        """
        錯誤處理：列印錯誤訊息、呼叫 log、解除 alarm。
        :param message: 自訂錯誤說明
        """
        print(f"❌ 發生錯誤：{message}")
        self.robot.log("activating alarm")
        self._reset_robot()
    

    def set_joint(self, rel=0, **kwargs):
        """
        嘗試移動機械手臂到指定角度位置。
        :param rel: 0 表示絕對角度，1 表示相對角度
        :param kwargs: 各關節角度 j0 ~ j4
        """
        try:
            # 設定移動指令
            result = self.robot.jmove(rel=rel, **kwargs)
            if result in [-400.0, -600.0]:
                self._handle_error("jmove 移動失敗")
        except Exception as e:
            print(f"例外錯誤：{e}")
            self._handle_error("執行 jmove 發生例外錯誤")

    def get_current_position(self):
        """
        回傳目前機械手臂的位置。
        :return: 各關節角度列表
        """
        try:
            pos = self.robot.get_joint()
            print("📍 目前位置：", pos)
            return pos
        except Exception as e:
            print(f"讀取位置錯誤：{e}")
            return None

    def move_to_initial(self):
        """
        回到預設初始位置。
        """
        self.set_joint(rel=0, **self.initial_joint_pos)
        
    def move_to_home(self):
        """
        回到移動開始位置。
        """
        self.set_joint(rel=0, **self.home_pos)

    def disconnect(self):
        """
        中斷與機械手臂的連線。
        """
        self.robot.close()
        print("🔌 已斷開連線")

'''
bot = RobotController('ROBOT_IP')
bot.set_joint(j1 = 170, j2 = -130)
bot._reset_robot()
bot.move_to_initial()
bot.move_to_home()
bot.disconnect()
robot.close()
'''