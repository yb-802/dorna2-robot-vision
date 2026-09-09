class Coordinator:
    '''
    camera、yolo整合
    '''
    def __init__(self, camera, yolo, processor):
        self.camera = camera
        self.yolo = yolo
        self.processor = processor

    def handle_frame(self, depth, color, intrinsics):
        results = self.yolo.capture_image(depth, color, intrinsics)
        self.processor.add_result(results)

    def start(self):
        self.camera.run(self.handle_frame)  # 讓 Camera 丟 frame，這裡接收並統一處理

