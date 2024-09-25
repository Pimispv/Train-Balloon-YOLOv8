# Train-Balloon-YOLOv8

Train object detection algorithm  for an especial class (Here Balloon Class) using YOLO v8

#### train balloon class ####

1- Put all pics in file and their labels in another file and put both these file into a file named: pretrain

2- Create config.yaml file including the address of these files into it.

3-type the following commands in the python environment or just copy them into a .py file and run it in cmd:


     from ultralytics import YOLO
     model = YOLO("yolov8n.yaml")  
     model.train(data="config.yaml", epochs=100)  


#### predict balloon class ####

1- just type this command: 

     python predict_cam.py
