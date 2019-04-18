# People-Counter
Download YOLOv3 or tiny_yolov3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).Then convert the Darknet YOLO model to a Keras model. Or use the converted model from https://drive.google.com/file/d/1uvXFacPnrSMw6ldWTyLLjGLETlEsUvcE/view?usp=sharing (yolo.h5 model file with tf-1.4.0) , then put it into model_data folder
Run yolo.py. I have used Anaconda cmd 
Run python demo.py
# Dependencies
NumPy
sklean
OpenCV
Pillow
Additionally, feature generation requires TensorFlow-1.4.0.
![image](https://github.com/tutul032/People-Counter/blob/master/people%20counter.jpg)
The code ignores everything but detecting only person.
