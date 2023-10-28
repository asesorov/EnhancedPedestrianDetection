from ultralytics import YOLO


class YoloNCNN:
    def __init__(self, ncnn_model_dir):
        self.load_model(ncnn_model_dir)

    def load_model(self, dir_path):
        self.model = YOLO(dir_path)

    def simple_predict(self, input):
        return self.model.predict(input, half=False)
