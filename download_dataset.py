from roboflow import Roboflow
rf = Roboflow(api_key="O8NgWA1xLYS553LPldiR")
project = rf.workspace("hcmute-vo4sj").project("shuttlecock-detection")
version = project.version(3)
dataset = version.download("yolov8")
