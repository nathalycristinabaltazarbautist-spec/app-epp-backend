from roboflow import Roboflow

rf = Roboflow(api_key="IJ4gG5iC6KlvDLZO2Cob")

project = rf.workspace("nathalys-workspace-mgloq").project("epp_detection-ke9vt")

version = project.version(1)
dataset = version.download("yolov8")