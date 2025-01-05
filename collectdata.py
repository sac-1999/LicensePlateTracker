from roboflow import Roboflow
import os 

rf = Roboflow(api_key=os.getenv('API_KEY'))
project = rf.workspace("kanwal-masroor-gv4jr").project("yolov7-license-plate-detection")
version = project.version(3)
dataset = version.download("yolov8")