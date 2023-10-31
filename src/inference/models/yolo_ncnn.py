from ultralytics import YOLO


class YoloNCNN:
    def __init__(self, ncnn_model_dir, task_type):
        self.load_model(ncnn_model_dir, task_type)

    def load_model(self, dir_path, task_type):
        """
        Load NCNN model through YOLO
        (task_type: 'detect', 'segment', 'classify', 'pose')
        """
        self.model = YOLO(dir_path, task=task_type)

    def simple_predict(self, input, **kwargs):
        """
        Returns YOLO.predict for NCNN inference
        """
        return self.model.predict(input, half=False)
