del /q c:\github\hyperbench\trainedCNN\tools\variables
rmdir c:\github\hyperbench\trainedCNN\tools\variables
del /q c:\github\hyperbench\trainedCNN\tools
rmdir c:\github\hyperbench\trainedCNN\tools
python retrain.py --image_dir .\tools_photos --output_graph=c:\github\hyperbench\trainedCNN\tools_recognizer.pb --bottleneck_dir=c:\github\hyperbench\trainedCNN\tools_bottleneck --output_labels=c:\github\hyperbench\trainedCNN\tools_labels.lst --flip_left_right --random_crop=10 --random_brightness=90