#inverse_kinematics
import math

class InverseKinematicsSolver:
    def __init__(self, L1=203.2, L2=152.4, L3=48.92, camera_offset=95.48, base_height=206.4, camera_z_offset=206.4):
        """
        初始化手臂參數與相機座標偏移。
        """
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.camera_offset = camera_offset
        self.base_height = base_height
        self.camera_z_offset = camera_z_offset

    def compute_angle_3dof(self, x, y, z, phi):
        """
        計算三關節機械手臂的角度。
        :param x: 實際座標系下的 x（向外）
        :param y: 實際座標系下的 y（向右）
        :param z: 實際座標系下的 z（向上）
        :param phi: 末端執行器姿態角（度，相對水平面）
        :return: (theta0, theta1, theta2, theta3)，單位為 degree
        """
        # 計算基座旋轉角
        theta0 = math.degrees(math.atan2(y, x))
        theta0 += 180
        
        # 將姿態角轉換為弧度
        phi_rad = math.radians(phi)
        
        # 計算手腕位置（第三關節的起點）
        # 投影到 XZ 平面的距離
        r_end = math.hypot(x, y)
        
        # 手腕在 XZ 平面的位置
        #r_wrist = r_end - self.L3 * math.cos(phi_rad)
        #z_wrist = z - self.L3 * math.sin(phi_rad)
        
        # 轉換至機械手臂座標系
        #x_wrist = r_wrist - self.camera_offset
        #z_wrist_adjusted = z_wrist - self.base_height
        
        # 檢查是否在工作範圍內
        #distance_to_wrist = math.hypot(x_wrist, z_wrist_adjusted)
        #if distance_to_wrist > (self.L1 + self.L2):
            #raise ValueError(f"手腕位置超出工作範圍: 距離={distance_to_wrist:.2f}, 最大距離={self.L1 + self.L2:.2f}")
        
        # 使用餘弦定理計算 theta2
        D = (self.L1**2 + self.L2**2 - x**2 - z**2) / (2 * self.L1 * self.L2)
        
        #if D < -1 or D > 1:
            #raise ValueError(f"手腕位置無解: D={D:.3f} (必須在[-1,1]範圍內)")
        
        # 計算 theta2 (肘關節)
        theta2_rad = math.acos(D) * (-1)
        theta2 = math.degrees(theta2_rad)
        
        # 計算 theta1 (肩關節)
        alpha = math.atan2(z, x)
        beta = math.atan2(self.L2 * math.sin(theta2_rad), self.L1 + self.L2 * math.cos(theta2_rad))
        theta1 = math.degrees(alpha - beta)
        
        # 計算 theta3 (腕關節)
        # theta3 使得末端執行器相對水平面的角度為 phi
        theta3 = phi - theta1 - theta2
        
        print(f"解1: theta1={theta1:.2f}°, theta2={theta2:.2f}°, theta3={theta3:.2f}°")
        
        # 第二個解（肘關節相反方向）
        theta2_rad_alt = math.acos((-1) * D) * (-1)
        theta2_alt = math.degrees(theta2_rad_alt)
        
        alpha_alt = math.atan2(z, x)
        beta_alt = math.atan2(self.L2 * math.sin(theta2_rad_alt), self.L1 + self.L2 * math.cos(theta2_rad_alt))
        theta1_alt = math.degrees(alpha_alt - beta_alt)
        
        theta3_alt = phi - theta1_alt - theta2_alt
        
        print(f"解2: theta1={theta1_alt:.2f}°, theta2={theta2_alt:.2f}°, theta3={theta3_alt:.2f}°")
        
        if abs(theta2) <= 142:
            if theta1 > theta1_alt:
                return theta0, theta1, theta2, theta3
            else:
                if abs(theta2_alt) <= 142:
                    return theta0, theta1_alt, theta2_alt, theta3_alt
                else:
                    return theta0, theta1, theta2, theta3
        elif abs(theta2_alt) <= 142:
            return theta0, theta1_alt, theta2_alt, theta3_alt
        else:
            print("超出")
                

if __name__ == "__main__":
    solver = InverseKinematicsSolver()
    
    # 範例：目標位置和姿態
    target_x = 1.924
    target_y = 0  
    target_z = 253.434
    target_phi = -107
    
    try:
        theta0, theta1, theta2, theta3 = solver.compute_angle_3dof(target_x, target_y, target_z, target_phi)
        print(f"\n最終結果:")
        print(f"θ0 (基座) = {theta0:.2f}°")
        print(f"θ1 (肩膀) = {theta1:.2f}°") 
        print(f"θ2 (肘部) = {theta2:.2f}°")
        print(f"θ3 (手腕) = {theta3:.2f}°")
        
    except ValueError as e:
        print(f"錯誤: {e}")
     


