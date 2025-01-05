from roboflow import Roboflow
rf = Roboflow(api_key="lu08WwdIJ89fRfTh16mT")
project = rf.workspace("kanwal-masroor-gv4jr").project("yolov7-license-plate-detection")
version = project.version(3)
dataset = version.download("yolov8")