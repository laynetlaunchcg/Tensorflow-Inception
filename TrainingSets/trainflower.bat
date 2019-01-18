del G:\Tensorflow-Inception\trainedCNN\flower\variables
rmdir G:\Tensorflow-Inception\trainedCNN\flower
del G:\Tensorflow-Inception\trainedCNN\flower
rmdir G:\Tensorflow-Inception\trainedCNN\flower

python retrain.py --image_dir .\flower_photos --saved_model_dir=G:\Tensorflow-Inception\trainedCNN\flower --output_graph=G:\Tensorflow-Inception\trainedCNN\flower_recognizer.pb --bottleneck_dir=G:\Tensorflow-Inception\trainedCNN\flower_bottleneck --output_labels=G:\Tensorflow-Inception\trainedCNN\flower_labels.txt

