del /q G:\Tensorflow-Inception\trainedCNN\tools\variables
rmdir G:\Tensorflow-Inception\trainedCNN\tools\variables
del /q G:\Tensorflow-Inception\trainedCNN\tools
rmdir G:\Tensorflow-Inception\trainedCNN\tools
python retrain.py --image_dir .\tools_photos --output_graph=G:\Tensorflow-Inception\trainedCNN\tools_recognizer.pb --bottleneck_dir=G:\Tensorflow-Inception\trainedCNN\tools_bottleneck --output_labels=G:\Tensorflow-Inception\trainedCNN\tools_labels.lst

REM --flip_left_right --random_crop=10 random_brightness=1.05