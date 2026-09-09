# Dorna 2 機械手臂智能操控與視覺整合系統 (dorna2-robot-vision)

本專案整合了 **Dorna 2 5軸機械手臂**、**Intel RealSense 深度相機**、**YOLOv7 物件偵測** 以及 **LLM 自然語言解析**，實現基於邊緣端/遠端協同運算之智能物件辨識與精準抓取操控。

---

## 📌 系統特點與架構

1. **LLM 語意解析 (LLM API Integration)**：
   - 使用者可以輸入自然語言指令（例如：「幫我拿鍵盤和滑鼠」或 "take the bottle"）。
   - 透過大語言模型自動提取目標物件，並映射至標準目標類別（如 COCO 80 類）。

2. **YOLOv7 視覺辨識與 3D 空間定位 (YOLOv7 & Depth Registration)**：
   - 使用 Intel RealSense 深度相機擷取 Color 與 Depth 影像並進行 Alignment 對齊。
   - 利用 YOLOv7 計算物件 2D 邊界框（Bounding Box）與中心像素座標 $(u, v)$。
   - 透過相機內參 (Intrinsics) 與深度資訊 $d$ 轉換為相機座標系下的三維座標 $(X_c, Y_c, Z_c)$。

3. **手眼校正與逆運動學 (Hand-Eye Calibration & Inverse Kinematics)**：
   - 結合座標轉換矩陣（包含旋轉 $R$ 與平移 $t$），將相機 3D 座標映射至 Dorna 2 機械手臂基座座標系 $(X_r, Y_r, Z_r)$。
   - 透過逆運動學 (IK) 解析幾何模型，精確計算出各關節角度 $(	heta_0, 	heta_1, 	heta_2, 	heta_3)$ 並驅動 Dorna 2。

---

## 📁 專案檔案結構 (Repository Structure)

```text
dorna2-robot-vision/
├── main.py                     # 系統主要進入點，協調視覺、LLM 與機械手臂運動
├── dorna_controller.py         # Dorna 2 機械手臂控制器介面
├── inverse_kinematics.py       # 逆運動學算法與關節角度解算
├── label_driven_motion.py      # 根據語意標籤驅動手臂路徑與抓取邏輯
├── llm_api.py                  # LLM API 串接與自然語言指令解析
├── realsense_camera.py         # Intel RealSense 深度相機讀取與對齊
├── vision_coordinator.py       # 視覺座標轉換與相機-手臂座標系整合
├── yolo_detector.py            # YOLOv7 物件偵測介面與推論
├── models/                     # YOLOv7 網路架構定義與模組
│   ├── common.py
│   ├── experimental.py
│   └── yolo.py
└── .gitignore.txt              # Git 忽略設定
```

---

## ⚙️ 環境安裝與需求

### 1. 硬體需求
- **機械手臂**: Dorna 2 5-Axis Robotic Arm
- **深度相機**: Intel RealSense D400 系列 (如 D435 / D435i)
- **運算平台**: 支持 CUDA 的 GPU 主機 / 遠端伺服器 (用於 YOLOv7 & LLM 解析)

### 2. 軟體依賴
建議使用 Python 3.9+ 環境：

```bash
pip install torch torchvision
pip install pyrealsense2 opencv-python numpy
pip install dorna
```

---

## 🚀 使用說明

1. **連接硬體設備**
   - 將 Intel RealSense 相機連接至 USB 3.0 埠。
   - 確保 Dorna 2 機械手臂電源開啟並已連接至區域網路/控制器。

2. **執行主程式**
   ```bash
   python main.py
   ```

3. **互動流程**
   - 程式會開啟 RealSense 實時畫面。
   - 在控制台輸入自然語言抓取指令。
   - 系統將自動執行：`LLM 標籤提取` ➔ `YOLOv7 定位` ➔ `3D 座標計算` ➔ `IK 關節角計算` ➔ `Dorna 2 抓取`。

---

## 📐 核心算法原理

### 1. 2D 像素轉相機 3D 座標
$$P_{camera} = 	ext{deproject\_pixel\_to\_point}(	ext{intrinsics}, [u, v], d)$$

### 2. 相機座標轉手臂座標
$$P_{robot} = R_2 \cdot R_1 \cdot P_{camera} + t$$

### 3. 逆運動學 (IK) 關節角求解
- **基座角度 $	heta_0$**: $	heta_0 =  rctan2(y, x)$
- **肘部關節 $	heta_2$**: 利用餘弦定理求解 $	heta_2 = - rccos\left(rac{x^2 + z^2 - L_1^2 - L_2^2}{2 L_1 L_2}
ight)$
- **肩部關節 $	heta_1$**: $	heta_1 =  rctan2(z, x) -  rctan2\left(rac{L_2 \sin	heta_2}{L_1 + L_2 \cos	heta_2}
ight)$
- **末端姿態角 $	heta_3$**: $	heta_3 = 	ext{target\_angle} - 	heta_1 - 	heta_2$

---

## 👤 開發資訊

- **專案名稱**: `dorna2-robot-vision`
- **主要語言**: Python
